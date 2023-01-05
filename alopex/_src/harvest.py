"""Module for the harvest transformation.

In short words, this is the reimplementation of `oryx.core` module.

`plant` replaces the output of specified `sow`-ed values with the dict of
first argument.

Example:
    sow and plant
    ```python
    from harvest import sow, plant

    f = lambda x: sow(2*x, tag="x", name="v")
    assert f(10) == 20
    assert plant(f, tag="x")({"v": 0}, 10) == 0
    ```

    sow and reap
    ```python
    from harvest import sow, reap

    f = lambda x: 2 * sow(x + 1, tag="x", name="v")
    assert f(9) == 20
    assert reap(f, tag="x")(9) == {"v": 10}
    ```
"""
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


def harvest(f: tp.Callable, *, tag: str) -> tp.Callable:
    """Apply harvest transformation to the function.

    Args:
        f: A function to transform. Note that it should not be jit-ed
            because harvest transformation requires f to be not-pure.
            To jit f, jit the harvest transformed function.
        tag: A tag of variable collection.

    Returns:
        Harvest transformed function.
    """

    def wrapped(plants: dict[str, tp.Any], *args, **kwargs):
        ctx_reaps = _thread_local.reaps = _get_dynamic_context("reaps")
        ctx_plants = _thread_local.plants = _get_dynamic_context("plants")

        if tag in ctx_reaps or tag in ctx_plants:
            raise RuntimeError(f"{tag} is already used. Use different tag.")

        ctx_reaps[tag] = {}
        ctx_plants[tag] = plants

        value = f(*args, **kwargs)

        # Remove `tag` values from ctx
        reaped = ctx_reaps.pop(tag)
        ctx_plants.pop(tag)

        return value, reaped

    return wrapped


def plant(f: tp.Callable, *, tag: str):
    def wrapped(plants: dict[str, chex.ArrayTree], *args, **kwargs):
        value, _ = harvest(f, tag=tag)(plants, *args, **kwargs)
        return value

    return wrapped


def call_and_reap(f: tp.Callable, *, tag: str) -> tp.Callable:
    """

    Returns:
        A transformed function that returns a tuple of (out, reaped).
    """

    def wrapped(*args, **kwargs):
        return harvest(f, tag=tag)({}, *args, **kwargs)

    return wrapped


def reap(f: tp.Callable, *, tag: str):
    """
    Returns:
        A transformed function that returns reaped.
    """
    return lambda *args, **kwargs: call_and_reap(f, tag=tag)(*args, **kwargs)[1]


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
