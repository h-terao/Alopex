import typing as tp
import functools


class Registry(dict):
    """A registry class.

    This is a registry object. Registry is a Python dict,
    so you can use dict methods like pop, merge, etc.
    """

    def register(self, name: str, fn: tp.Callable | None = None, **kwargs) -> tp.Callable:
        """Register a callable object in this registry.

        Args:
            name: Name of a callable object to register.
            fn: A callable object to register.
                If None, this method works as a decorator.
            **kwargs: Default values of a callable object. Useful to register
                callable with different parameters.

        Raises:
            If name is already registered, raise an error.
        """
        if name in self:
            raise RuntimeError(f"{name} is already registered.")

        if fn is None:
            # decorator.
            def deco(f):
                self[name] = functools.partial(f, **kwargs)
                return fn

            return deco
        else:
            new_fn = functools.partial(fn, **kwargs)
            self[name] = new_fn
            return new_fn


# Create a global registry for the simple usage.
registry = Registry()
register = registry.register
