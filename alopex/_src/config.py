from __future__ import annotations
import typing as tp
import functools
import threading


local_config = threading.local()


class _Unspecified:
    """An object class used as the default value of `get_config`."""

    pass


unspecified = _Unspecified()


def set_config(fun: tp.Callable, mapping: dict[str, tp.Any] | None = None, **kwargs) -> tp.Callable:
    """
    Args:
        fun: A function that access the configurations inside.

    Returns:
        A wrapped version of `fun`. In the transformed function, the configured values
        are available via `get_config(name)` method.

    Example:
        ::

            fun = lambda: print(get_config("train_mode", default="unspecified"))
            fun()  # print `unspecified`.
            set_config(fun, train_mode=False)  # print `false`
            get_config("train_mode", "x")  # print `x`
    """
    if mapping is None:
        mapping = {}

    configs = dict(mapping, **kwargs)
    for name in configs:
        if not hasattr(local_config, name):
            setattr(local_config, name, list())

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        for name, value in configs.items():
            getattr(local_config, name).append(value)

        try:
            outputs = fun(*args, **kwargs)
        finally:
            for name, value in reversed(configs.items()):
                getattr(local_config, name).pop()  # remove.

        return outputs

    return wrapped


def get_config(name: str, default: tp.Any = unspecified) -> tp.Any:
    if hasattr(local_config, name) and getattr(local_config, name):
        return getattr(local_config, name)[-1]
    elif not isinstance(default, _Unspecified):
        return default
    else:
        raise RuntimeError(f"{name} is not configured for this function.")
