""""""
from __future__ import annotations
import typing as tp
import warnings

import jax
import jax.numpy as jnp
from jax import tree_util
import chex

from .pytypes import (
    Batch,
    Summary,
    Prediction,
    TrainState,
    TrainFn,
    EvalFn,
    PredFn,
)

__all__ = ["train_epoch", "eval_epoch", "predict_epoch"]


def _replicate(tree, devices=None):
    # Modify flax.jax_util
    devices = devices or jax.local_devices()
    return jax.device_put__replicated(tree, devices)


def _unreplicate(tree):
    # from flax.jax_util
    return tree_util.tree_map(lambda x: x[0], tree)


def _modify_batches(
    iterable: tp.Iterable,
    epoch_length: int = -1,
    prefetch: bool = False,
    devices: list[chex.Device] | None = None,
):
    devices = devices or jax.local_devices()

    def modify_batch(batch: Batch) -> list[tuple[Batch, float]]:
        num_devices = len(devices)

        leaves, treedef = tree_util.tree_flatten(batch)
        min_size = min(len(x) for x in leaves)
        min_main_size, min_sub_size = divmod(min_size, num_devices)

        main_leaves, sub_leaves = [], []
        for x in leaves:
            r = len(x) / min_size
            main_size, sub_size = int(r * min_main_size), int(r * min_sub_size)
            assert main_size * num_devices + sub_size == len(x)

            if main_size > 0:
                main_array = x[sub_size:]
                main_leaves.append(main_array.reshape(num_devices, main_size, *x.shape[1:]))

            if sub_size > 0:
                sub_array = x[:sub_size]
                sub_leaves.append(jnp.stack([sub_array] * num_devices, axis=0))

        modified_batches = []
        if main_leaves:
            main_batch = tree_util.tree_unflatten(treedef, main_leaves)
            modified_batches.append([main_batch, main_size * num_devices, False])

        if sub_leaves:
            sub_batch = tree_util.tree_unflatten(treedef, sub_leaves)
            modified_batches.append([sub_batch, sub_size, True])

        return modified_batches

    prev_batch, prev_size, prev_is_remainder = None, None, None
    for batch_idx, batch in enumerate(iterable):
        modified_batches = modify_batch(batch)
        for next_batch, next_size, is_reminder in modified_batches:
            if prefetch:
                next_batch = tree_util.tree_map(
                    lambda x: jax.device_put_sharded(list(x), devices),
                    tree=next_batch,
                )

            if prev_batch is not None:
                yield prev_batch, prev_size, prev_is_remainder

            prev_batch, prev_size, prev_is_remainder = next_batch, next_size, is_reminder

        if batch_idx + 1 == epoch_length:
            break

    if prev_batch is not None:
        yield prev_batch, prev_size, prev_is_remainder


@jax.jit
def _accumulate_scalars(
    accum_scalars: dict[str, tuple[float, float]],
    new_scalars: dict[str, float],
    weight: float = 1,
) -> dict[str, tuple[float, float]]:
    updates = {}
    for key, scalar in new_scalars.items():
        accum_scalar, accum_weight = accum_scalars.get(key, (0, 0))
        scalar = jnp.array(scalar, dtype=jnp.float32)  # If scalar is float16, overflow may cause.
        accum_scalar += scalar.mean() * weight
        accum_weight += weight
        updates[key] = (accum_scalar, accum_weight)
    return dict(accum_scalars, **updates)


def _summarize_scalars(
    prefix: str, accum_scalars: dict[str, tuple[float, float]], **kwargs
) -> dict[str, float]:
    summary = {prefix + key: float(val / weight) for key, (val, weight) in accum_scalars.items()}
    for key, val in kwargs.items():
        summary[prefix + key] = val
    return summary


def train_epoch(
    train_state: TrainState,
    iterable: tp.Iterable,
    train_fn: TrainFn,
    epoch_length: int = -1,
    prefix: str | None = None,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> tuple[TrainState, Summary]:
    """Train the model on the given dataset.

    Args:
        train_state: Flax train state.
        iterable: Data loader that yields batches.
        train_fn: Training function that updates train_state.
        epoch_length: If the number of batches reaches this value, stop training.
        prefix: Prefix of scalars.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device list. If None, all visible devices are used.

    Returns:
        Updated train_state and average metrics.

    """
    num_devices = len(devices or jax.local_devices())
    prefix = prefix or ""

    p_train_fn = jax.pmap(train_fn, axis_name=axis_name, devices=devices)
    train_state = _replicate(train_state, devices)

    accum_scalars = {}
    for batch, weight, is_remainder in _modify_batches(
        iterable=iterable, epoch_length=epoch_length, prefetch=prefetch, devices=devices
    ):
        if is_remainder:
            msg = (
                "You set batch size that is not divisible by the number of devices "
                f"(#devices={num_devices}). This configuration will perform inefficient "
                "and not expected training behaviour. "
            )
            warnings.warn(msg)

        train_state, scalars = p_train_fn(train_state, batch)
        accum_scalars = _accumulate_scalars(accum_scalars, scalars, weight)

    train_state = _unreplicate(train_state)
    summary = _summarize_scalars(prefix, accum_scalars)
    return train_state, summary


def eval_epoch(
    train_state: TrainState,
    iterable: tp.Iterable,
    eval_fn: EvalFn,
    epoch_length: int = -1,
    prefix: str | None = None,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> Summary:
    """Evaluate the model on the given dataset.

    Args:
        train_state: Flax train state.
        iterable: Data loader that yields batches.
        eval_fn: Evaluation function that returns metrics.
        epoch_length: If the number of batches reaches this value, stop evaluating.
        prefix: Prefix of scalars.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device to use. If None, all visible devices are used.

    Returns:
        Average metrics.
    """
    prefix = prefix or ""

    p_eval_fn = jax.pmap(eval_fn, axis_name=axis_name, devices=devices)
    train_state = _replicate(train_state, devices)

    accum_scalars = {}
    for batch, weight, _ in _modify_batches(
        iterable=iterable, epoch_length=epoch_length, prefetch=prefetch, devices=devices
    ):
        scalars = p_eval_fn(train_state, batch)
        accum_scalars = _accumulate_scalars(accum_scalars, scalars, weight)

    summary = _summarize_scalars(prefix, accum_scalars)
    return summary


def predict_epoch(
    train_state: TrainState,
    iterable: tp.Iterable,
    predict_fn: PredFn,
    epoch_length: int = -1,
    prefix: str | None = None,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> Prediction:
    """Stacking output PyTrees of predict_fn. Maybe useful to experiment generative models.

    Args:
        train_state: Flax train state.
        iterable: Data loader that yields batches.
        eval_fn: Evaluation function that returns metrics.
        epoch_length: If the number of batches reaches this value, stop evaluating.
        prefix: Prefix of scalars.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device to use. If None, all visible devices are used.

    Returns:
        Average metrics.
    """
    prefix = prefix or ""

    p_pred_fn = jax.pmap(predict_fn, axis_name=axis_name, devices=devices)
    train_state = _replicate(train_state, devices)

    outputs = []
    for batch, _, is_remainder in _modify_batches(
        iterable=iterable, epoch_length=epoch_length, prefetch=prefetch, devices=devices
    ):
        output = p_pred_fn(train_state, batch)
        if is_remainder:
            output = tree_util.tree_map(lambda x: x[0], output)
        else:
            output = tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), output)
        outputs.append(output)

    output = tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)
    return output
