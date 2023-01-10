"""Reimplementation of the harvest transformation from `oryx.core`"""
from __future__ import annotations
import typing as tp
import threading

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


def sow(value: chex.ArrayTree, *, tag: str, name: str, mode: str = "strict") -> chex.Array:
    """Tag values in a function if `sow` is called in a function transformed by
        harvest. Otherwise, this function performs as an identity function.

    Args:
        value: Array value.
        tag: Tag name.
        name: Value name.
        mode: strict, clobber or append. If strict, raise an error if (tag, name) is already
            registered.

    Returns:
        Tagged value.
    """
    ctx_reaps = _get_dynamic_context("reaps")
    if tag in ctx_reaps:
        if mode in ["strict", "clobber"]:
            if mode == "strict" and name in ctx_reaps[tag]:
                msg = (
                    f"strict mode is specified but (tag, name)=({tag}, {name}) "
                    "is already registered. "
                )
                raise RuntimeError(msg)
            ctx_reaps[tag].setdefault(name, {})
            ctx_reaps[tag][name] = value
        elif mode == "append":
            ctx_reaps[tag].setdefault(name, tuple())
            ctx_reaps[tag][name] += (value,)
        else:
            raise ValueError(f"Unknown mode ({mode}) is specified.")

    ctx_plants = _get_dynamic_context("plants")
    if tag in ctx_plants:
        if name in ctx_plants[tag]:
            value = ctx_plants[tag][name]

    return value


def sow_grad(x: chex.Array, tag: str = "grad", *, name: str) -> chex.Array:
    """Tag values to take their gradients.

    Tag values inside a function. The gradients of tagged arrays can be collected via `reap` method.
    Note that `reap` should wrap the grad function to obtain gradients.
    This function is useful to obtain grads of intermediate values.

    Args:
        x: Array to take a grads.
        tag: Tag name of grads.
        name: Name of `x`.

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
        return x, x

    def backward(x, dy):
        dy = sow(dy, tag=tag, name=name)
        _, vjp = jax.vjp(forward, x)
        return vjp(dy)

    identity.defvjp(forward, backward)
    return identity(x)


def harvest(fun: tp.Callable, *, tag: str) -> tp.Callable:
    """Creates a function that harvest sow-ed values in fun.

    Args:
        fun: Function to harvest.
        tag: A tag of variable collection.

    Returns:
        A wraped version of fun.

    NOTE:
        You cannot directly jit fun. If you'd like to jit fun,
        jit the wrapped version of fun.
    """

    def wrapped(plants: dict[str, tp.Any], *args, **kwargs):
        ctx_reaps = _thread_local.reaps = _get_dynamic_context("reaps")
        ctx_plants = _thread_local.plants = _get_dynamic_context("plants")

        if tag in ctx_reaps or tag in ctx_plants:
            raise RuntimeError(f"{tag} is already used. Use different tag.")

        ctx_reaps[tag] = {}
        ctx_plants[tag] = plants

        value = fun(*args, **kwargs)

        # Remove `tag` values from ctx
        reaped = ctx_reaps.pop(tag)
        ctx_plants.pop(tag)

        return value, reaped

    return wrapped


def plant(fun: tp.Callable, *, tag: str) -> tp.Callable:
    """Creates a function that replaces sow-ed values in fun to the specified `plants`.

    Args:
        fun: Function to plant values.
        tag: A tag of value collection.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(plants: dict[str, chex.ArrayTree], *args, **kwargs):
        value, _ = harvest(fun, tag=tag)(plants, *args, **kwargs)
        return value

    return wrapped


def call_and_reap(fun: tp.Callable, *, tag: str) -> tp.Callable:
    """Creates a function that returns outputs and collection of sow-ed values from fun.

    Args:
        fun: Function to collect sow-ed values.
        tag: A tag of value collection.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(*args, **kwargs):
        return harvest(fun, tag=tag)({}, *args, **kwargs)

    return wrapped


def reap(fun: tp.Callable, *, tag: str):
    """Creates a function that returns outputs and collection of sow-ed values from fun.

    Args:
        fun: Function to collect sow-ed values.
        tag: A tag of value collection.

    Returns:
        A wrapped version of fun.
    """
    return lambda *args, **kwargs: call_and_reap(fun, tag=tag)(*args, **kwargs)[1]


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    def f(x):
        print("compile...")
        y = sow(x**2, tag="x", name="squared")
        z = sow(2 * x, tag="x", name="twice")
        return y, z

    reap_jit = jax.vmap(reap(f, tag="x"))
    print(reap_jit(jnp.arange(10)))

    reap_jit = jax.vmap(plant(f, tag="x"))
    print(reap_jit({"twice": 7 * jnp.ones(10)}, jnp.arange(10)))
