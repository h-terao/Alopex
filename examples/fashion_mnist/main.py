from __future__ import annotations
import typing as tp
from pathlib import Path
from functools import partial
import math
import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
from flax import linen
import optax
import alopex as ap
import chex

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.experimental.set_visible_devices([], "GPU")


class MLP(linen.Module):
    hidden_size: int
    dropout: float = 0.5
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        batch_size = len(x)
        x = x.reshape(batch_size, -1)

        x = linen.Dense(self.hidden_size)(x)
        x = linen.BatchNorm(not is_training, axis_name=self.axis_name)(x)
        x = linen.relu(x)

        x = linen.Dense(self.hidden_size)(x)
        x = linen.BatchNorm(not is_training, axis_name=self.axis_name)(x)
        x = linen.relu(x)

        x = linen.Dropout(self.dropout)(x, not is_training)
        x = linen.Dense(10)(x)

        return x


class TrainState(tp.NamedTuple):
    step: int
    rng: chex.PRNGKey
    params: chex.ArrayTree
    state: chex.ArrayTree
    opt_state: optax.OptState


def make_model(
    hidden_size: int = 128,
    dropout: float = 0.5,
    learning_rate: float = 0.1,
    weight_decay: float = 0,
    axis_name: str | None = None,
):
    model = MLP(hidden_size, dropout, axis_name=axis_name)
    optimizer = optax.chain(
        optax.add_decayed_weights(
            weight_decay, lambda params: tree_util.tree_map(lambda p: p.ndim > 1, params)
        ),
        optax.sgd(learning_rate, momentum=0.9),
    )

    def init_fun(rng, batch):
        inputs, _ = batch
        rng, init_rng = jr.split(rng)

        variables = model.init(init_rng, inputs)
        state, params = variables.pop("params")
        opt_state = optimizer.init(params)

        return TrainState(step=0, rng=rng, params=params, state=state, opt_state=opt_state)

    def apply_fun(rng, params, state, inputs, is_training=False):
        """Haiku-like API."""
        if rng is not None:
            rng = {"dropout": rng}
        variables = {"params": params, **state}
        output, new_state = model.apply(
            variables, inputs, is_training, rngs=rng, mutable="batch_stats"
        )
        return output, new_state

    @partial(ap.train_epoch, prefix="train/", axis_name=axis_name)
    def train_fun(train_state, batch):
        inputs, labels = batch
        step, rng, params, state, opt_state = train_state

        rng, new_rng = jr.split(rng)
        rng = jr.fold_in(rng, jax.lax.axis_index(axis_name))

        # Preprocessing
        rng, flip_rng = jr.split(rng)
        inputs = jnp.where(jr.uniform(flip_rng) < 0.5, inputs, inputs[:, :, ::-1])
        inputs = inputs / 255
        labels = linen.one_hot(labels, 10)

        @partial(jax.grad, has_aux=True)
        def grad_fun(params):
            logits, new_state = apply_fun(rng, params, state, inputs, is_training=True)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            scalars = {"loss": loss, "acc": ap.accuracy(logits, labels).mean()}
            return loss, (scalars, new_state)

        grads, (scalars, new_state) = grad_fun(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_train_state = TrainState(step + 1, new_rng, new_params, new_state, new_opt_state)
        return new_train_state, scalars

    @partial(ap.eval_epoch, prefix="val/", axis_name=axis_name)
    def val_fun(train_state, batch):
        inputs, labels = batch
        inputs = inputs / 255.0
        labels = linen.one_hot(labels, 10)
        logits, _ = apply_fun(None, train_state.params, train_state.state, inputs)
        return {
            "loss": optax.softmax_cross_entropy(logits, labels).mean(),
            "acc": ap.accuracy(logits, labels).mean(),
        }

    return init_fun, train_fun, val_fun


def make_loaders(batch_size):
    ds: tp.Mapping[str, tf.data.Dataset] = tfds.load("fashion_mnist", as_supervised=True)
    train_data = (
        ds["train"]
        .cache()
        .repeat()
        .shuffle(buffer_size=60_000)
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )
    test_data = ds["test"].batch(batch_size).cache().repeat().as_numpy_iterator()
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory.")
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed value.")
    parser.add_argument("--epochs", "-e", type=int, default=64, help="Number of epochs.")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--drouput", type=float, default=0, help="Dropout ratio.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0, help="Wegit decay rate.")
    args = parser.parse_args()

    rng = jr.PRNGKey(args.seed)
    init_fun, train_fun, val_fun = make_model(
        args.hidden_size,
        args.drouput,
        args.learning_rate,
        args.weight_decay,
        axis_name="batch",
    )

    # Setup alopex loggers.
    logger = ap.LoggerCollection(
        ap.ConsoleLogger(),
        ap.DiskLogger(args.out_dir),
    )

    # Setup data loaders.
    train_loader, test_loader = make_loaders(args.batch_size)

    # Initialize model and optimizer.
    train_state = init_fun(rng, next(train_loader))

    # Start training.
    for epoch in range(1, args.epochs + 1):
        # Train and evaluation.
        train_state, summary = train_fun(
            train_state, train_loader, epoch_length=60_000 // args.batch_size
        )
        summary |= val_fun(
            train_state, test_loader, epoch_length=math.ceil(10_000 / args.batch_size)
        )

        # Logging summary.
        logger.log_summary(summary, step=int(train_state.step), epoch=epoch)

    ap.plot_log_on_disk(
        Path(args.out_dir, "loss.png"),
        Path(args.out_dir, "log.json"),
        y_keys=["train/loss", "val/loss"],
        x_key="epoch",
        xlabel="Epoch",
        ylabel="Loss",
        title="Cross-entropy loss",
    )

    ap.plot_log_on_disk(
        Path(args.out_dir, "accuracy.png"),
        Path(args.out_dir, "log.json"),
        y_keys=["train/acc", "val/acc"],
        x_key="epoch",
        xlabel="Epoch",
        ylabel="Accuracy [%]",
        ylim=[0, 1.05],
        title="Accuracy",
    )


if __name__ == "__main__":
    main()
