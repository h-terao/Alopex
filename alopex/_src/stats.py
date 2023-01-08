"""Compute stats of models and functions.

MEMO:
    - Instead of directly returning stats, transform function that returns stats like harvest?
"""
from __future__ import annotations
import typing as tp
import math
import time

import numpy as np
import jax
from jax import tree_util
import chex


def _convert_size(v: chex.Scalar, unit: str | None = None) -> chex.Scalar:
    units = [None, "K", "M", "G", "T", "P", "E", "Z"]
    unit = None if unit is None else unit.upper()
    assert unit in units, f"Invalid unit is specified. Use {units}."

    idx = units.index(unit)
    return v / math.pow(1000, idx)


def count_flops(fn: tp.Callable, *args, unit: str | None = None, **kwargs) -> chex.Scalar:
    """Compute FLOPs of a function.

    Modify from:
      https://github.com/google-research/scenic/blob/main/scenic/common_lib/debug_utils.py

    Args:
        fn: A function to compute FLOPs.
        unit: A unit of FLOPs.
            "K", "M", "G", "T", "P", "E" and "Z" are available.
        *args, **kwargs: Function arguments.

    Returns:
        FLOPs of function.
    """
    computation = jax.xla_computation(fn)(*args, **kwargs)
    module = computation.as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
    flops = analysis["flops"]
    return _convert_size(flops, unit)


def count_macs(
    fn: tp.Callable,
    *args,
    unit: str | None = None,
    **kwargs,
) -> chex.Scalar:
    """Compute MACs of a function. This is more commonly used than FLOPs in literature.

    Args:
        fn: A function to count MACs.
        unit: A unit of MACs.
            "K", "M", "G", "T", "P", "E" and "Z" are available.
        *args, **kwargs: Function arguments.

    Returns:
        MACs of function.
    """
    return count_flops(fn, *args, unit=unit, **kwargs) / 2


def count_params(tree: chex.ArrayTree, unit: str | None = None) -> chex.Scalar:
    """Count number of elements stored in PyTree.

    Args:
        tree: A PyTree to count elements.
        unit: A unit of number of parameters.
            "K", "M", "G", "T", "P", "E" and "Z" are available.

    Returns:
        Number of elements stored in tree.
    """
    tree_size = sum([x.size for x in tree_util.tree_leaves(tree)])
    return _convert_size(tree_size, unit)


class TimeStats(tp.NamedTuple):
    avg: float
    std: float
    max: float
    min: float
    median: float


def time_fn(fn: tp.Callable, *args, n: int = 100, warmup: int = 0, **kwargs) -> TimeStats:
    """Time a function.

    Args:
        fn: A function to time.
        n: Number of steps to time.
        warmup: Number of warmup steps.
        *args, **kwargs: Arguments of `f`.

    Returns:
        A namedtuple of avg, std, max, min and median of elapsed times to run fn.
    """

    def call():
        start_time = time.time()
        jax.block_until_ready(fn(*args, **kwargs))
        return time.time() - start_time

    for _ in range(warmup):
        call()

    x = np.array([call() for _ in range(n)])
    return TimeStats(
        avg=np.mean(x),
        std=np.std(x),
        max=np.max(x),
        min=np.min(x),
        median=np.std(x),
    )
