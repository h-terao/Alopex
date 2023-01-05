from __future__ import annotations
import typing as tp
import logging

from flax import traverse_util


def maybe_load_arrays(tree: tp.Mapping, to_load: tp.Mapping, verbose: bool = False):
    def warn(msg):
        if verbose:
            logging.warn(msg)

    tree_flat = traverse_util.flatten_dict(tree)
    to_load_flat = traverse_util.flatten_dict(to_load)

    xs = {}
    for key, value in tree_flat.items():
        if key in to_load_flat:
            new_value = to_load_flat[key]
            if value.shape == new_value.shape:
                value = new_value
            else:
                warn(
                    (
                        f"Mismatch the shape of {key} ({value.shape} != {new_value.shape}). "
                        "This array is not updated."
                    )
                )
        else:
            warn(f"{key} is not found in to_load. This array is not updated.")
        xs[key] = value

    xs = traverse_util.unflatten_dict(xs)
    return xs
