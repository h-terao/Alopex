"""Reimplementation of the harvest transformation from `oryx.core`"""
from __future__ import annotations
import typing as tp
from functools import wraps
import threading

import jax
import chex

__all__ = [
    "sow",
    "harvest",
    "plant",
    "call_and_reap",
    "reap",
]


_thread_local = threading.local()


def _get_dynamic_context(name: str) -> dict:
    return getattr(_thread_local, name, dict())


def sow(value: chex.ArrayTree, *, col: str, name: str, mode: str = "strict", reverse: bool = False) -> chex.Array:
    """Tag values in a function if `sow` is called in a function transformed by
        harvest. Otherwise, this function performs as an identity function.

    Args:
        value: Array value.
        col: Tag name.
        name: Value name.
        mode: strict, clobber or append. If strict, raise an error if (col, name) is already
            registered.
        reverse: Only used if mode is append.

    Returns:
        Tagged value.
    """
    ctx_reaps = _get_dynamic_context("reaps")
    if col in ctx_reaps:
        if mode in ["strict", "clobber"]:
            if mode == "strict" and name in ctx_reaps[col]:
                msg = (
                    f"strict mode is specified but (col, name)=({col}, {name}) "
                    "is already registered. "
                )
                raise RuntimeError(msg)
            ctx_reaps[col].setdefault(name, {})
            ctx_reaps[col][name] = value
        elif mode == "append":
            ctx_reaps[col].setdefault(name, tuple())
            if reverse:
                ctx_reaps[col][name] = (value,) + ctx_reaps[col][name] + (value,)
            else:
                ctx_reaps[col][name] = ctx_reaps[col][name]
        else:
            raise ValueError(f"Unknown mode ({mode}) is specified.")

    ctx_plants = _get_dynamic_context("plants")
    if col in ctx_plants:
        if name in ctx_plants[col]:
            value = ctx_plants[col][name]

    return value


def sow_grad(x: chex.Array, col: str = "grad", *, name: str, mode: str = "strict", reverse: bool = False) -> chex.Array:
    """Tag values to take their gradients.

    Tag values inside a function. The gradients of colged arrays can be collected via `reap` method.
    Note that `reap` should wrap the grad function to obtain gradients.
    This function is useful to obtain grads of intermediate values.

    Args:
        x: Array to take a grads.
        col: Tag name of grads.
        name: Name of `x`.
        mode: Mode.

    Returns:
        `x` itself.

    Example:
        >>> f = lambda x: 2 * sow_grad(jnp.sin(x), name="x2")
        >>> f = reap(jax.grad(f))
        >>> assert f(1.0) == {"x2": ...}
    """

    @jax.custom_vjp
    def identity(x):
        return x

    def forward(x):
        return x, ()

    def backward(res, g):
        g = sow(g, col=col, name=name, mode=mode, reverse=not reverse)
        return (g,) 
        
    identity.defvjp(forward, backward)
    return identity(x)


def harvest(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that harvest sow-ed values in fun.

    Args:
        fun: Function to harvest.
        col: Name of variable collection to reap.

    Returns:
        A wraped version of fun.

    NOTE:
        You cannot directly jit fun. If you'd like to jit fun,
        jit the wrapped version of fun.
    """

    def wrapped(plants: dict[str, tp.Any], *args, **kwargs):
        ctx_reaps = _thread_local.reaps = _get_dynamic_context("reaps")
        ctx_plants = _thread_local.plants = _get_dynamic_context("plants")

        if col in ctx_reaps or col in ctx_plants:
            raise RuntimeError(f"{col} is already used. Use different name.")

        ctx_reaps[col] = {}
        ctx_plants[col] = plants

        value = fun(*args, **kwargs)

        reaped = ctx_reaps.pop(col)
        ctx_plants.pop(col)

        return value, reaped

    return wrapped


def plant(fun: tp.Callable, *, col: str | tp.Sequence[str]) -> tp.Callable:
    """Creates a function that replaces sow-ed values in fun to the specified `plants`.

    Args:
        fun: Function to plant values.
        col: A col of value collection.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(plants: dict[str, chex.ArrayTree], *args, **kwargs):
        value, _ = harvest(fun, col=col)(plants, *args, **kwargs)
        return value

    return wrapped


def call_and_reap(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that returns outputs and collection of sow-ed values from fun.

    Args:
        fun: Function to collect sow-ed values.
        col: A name of variable collection to reap.

    Returns:
        A wrapped version of fun.
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return harvest(fun, col=col)({}, *args, **kwargs)

    return wrapped


def reap(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that returns a collection of sow-ed values from fun.

    Args:
        fun: Function to collect sow-ed values.
        col: A name of variable collection to reap.

    Returns:
        A wrapped version of fun.
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        _, reaped = call_and_reap(fun, col=col)(*args, **kwargs)
        return reaped

    return wrapped


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    def f(x):
        print("compile...")
        y = sow(x**2, col="x", name="squared")
        z = sow(2 * x, col="x", name="twice")
        return y, z

    reap_jit = jax.vmap(reap(f, col="x"))
    print(reap_jit(jnp.arange(10)))

    reap_jit = jax.vmap(plant(f, col="x"))
    print(reap_jit({"twice": 7 * jnp.ones(10)}, jnp.arange(10)))
