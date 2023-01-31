from __future__ import annotations
import typing as tp

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


def make_padding(
    kernel_size: int | tp.Sequence[int],
    stride: int | tp.Sequence[int] = 1,
    dilation: int | tp.Sequence[int] = 1,
    num_spatial_dims: int | None = None,
) -> tp.Sequence[tuple[int, int]]:
    """Creates a PyTorch-like padding parameter from kernel_size.

    Args:
        kernel_size: Kernel size of convolution.
        stride: Stride parameter.
        dilation: Dilation parameter.
        num_spatial_dims: Number of kernel dimensions. This argument is required when
            kernel_size is an integer. If kernel_size is a sequence of int, this argument
            is ignored and len(kernel_size) is used as num_spatial_dims.

    Returns:
        Padding parameter.
    """
    if isinstance(kernel_size, int):
        msg = "If kernel_size is an integer, specify num_spatial_dims."
        assert num_spatial_dims is not None, msg
        kernel_size = [kernel_size] * num_spatial_dims

    num_spatial_dims = num_spatial_dims or len(kernel_size)
    assert len(kernel_size) == num_spatial_dims

    if isinstance(stride, int):
        stride = (stride,) * num_spatial_dims
    assert len(stride) == num_spatial_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_spatial_dims
    assert len(dilation) == num_spatial_dims

    padding = []
    for k, s, d in zip(kernel_size, stride, dilation):
        pad_size = ((s - 1) + d * (k - 1)) // 2
        padding.append((pad_size, pad_size))

    return padding
