"""Model and function stats."""
from __future__ import annotations
import typing as tp
import time as time_module

import numpy
import jax
from jax import tree_util
import chex


def count_flops(
    f: tp.Callable,
    *args,
    static_argnums: int | tp.Sequence[int] = (),
    **kwargs,
) -> int:
    """Compute FLOPs of apply_fn.

    Modify from:
      https://github.com/google-research/scenic/blob/main/scenic/common_lib/debug_utils.py

    Args:
        apply_fn: A function to compute FLOPs.
        fuse_multiply_add: If true, count a multiply and add (also known as
            "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
            HLO analysis). This is commonly used in literature.
        *args, **kwargs: Inputs of `apply_fn`.

    Returns:
        FLOPs of `apply_fun`
    """
    computation = jax.xla_computation(f, static_argnums=static_argnums)(*args, **kwargs)
    module = computation.as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
    flops = analysis["flops"]
    return flops


def count_macs(
    f: tp.Callable,
    *args,
    static_argnums: int | tp.Sequence[int] = (),
    **kwargs,
):
    """Compute MACs of a function.

    Modify from:
      https://github.com/google-research/scenic/blob/main/scenic/common_lib/debug_utils.py

    Args:
        f: A function to count MACs.
        fuse_multiply_add: If true, count a multiply and add (also known as
            "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
            HLO analysis). This is commonly used in literature.
        *args, **kwargs: Inputs of `apply_fn`.

    Returns:
        FLOPs of `apply_fun`
    """
    flops = count_flops(f, *args, static_argnums=static_argnums, **kwargs)
    return flops / 2


def count_params(tree: chex.ArrayTree) -> int:
    """Count number of elements stored in PyTree.

    Args:
        tree: A PyTree to count elements.

    Returns:
        Number of elements stored in tree.
    """
    return sum([x.size for x in tree_util.tree_leaves(tree)])


def time(f: tp.Callable, *args, n: int = 100, warmup: int = 0, **kwargs) -> dict[str, float]:
    """Time a function.

    Args:
        f: A function to time.
        n: Number of steps to time.
        warmup: Number of warmup steps.
        *args, **kwargs: Arguments of `f`.

    Returns:
        A dict that contains avg, std, max, min and
        median of elapsed time.
    """

    def call():
        start_time = time_module.time()
        jax.block_until_ready(f(*args, **kwargs))
        return time_module.time() - start_time

    for _ in range(warmup):
        call()

    x = numpy.array([call() for _ in range(n)])
    return {
        "avg": numpy.mean(x),
        "std": numpy.std(x),
        "max": numpy.max(x),
        "min": numpy.min(x),
        "median": numpy.median(x),
    }
