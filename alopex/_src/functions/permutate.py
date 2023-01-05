import jax.numpy as jnp
import chex


def permutate(
    inputs: chex.Array, index: chex.Array, axis: int = 0, inv: bool = False
) -> chex.Array:
    """Permutate array according to the given index.

    Args:
        inputs: Array with shape (N, ...).
        index: Index array that has a shape of (N).
        inv: If True, inverse the permutation operation.
            It means that x == permutate(permutate(x, index), index, inv=True).

    Returns:
        Permutated array.
    """
    assert len(index) == inputs.shape[axis]
    ndim = inputs.ndim
    axes = [axis] + [x for x in range(ndim) if x != axis]
    inputs = jnp.transpose(inputs, axes)

    if inv:
        index = jnp.zeros_like(index).at[index].set(jnp.arange(len(index)))
    out = jnp.take(inputs, index, axis=0)

    out = jnp.transpose(out, list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim)))
    return out
