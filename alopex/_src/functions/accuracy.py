import jax.numpy as jnp
import chex


def accuracy(inputs: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.

    Args:
        inputs: Prediction with shape (..., num_classes).
        labels: One-hot encoded labels with shape (..., num_classes).

    Returns:
        An binary array with shape (...). You can obtain accuracy
        via `accuracy(...).mean()`
    """
    assert inputs.shape == labels.shape
    y = jnp.argsort(inputs)[..., -k:]
    t = jnp.argmax(labels, axis=-1, keepdims=True)
    return jnp.sum(y == t, axis=-1)
