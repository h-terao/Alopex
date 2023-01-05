import functools


class Registry(dict):
    """A registry class.

    This is a registry object. Registry is a Python dict,
    so you can use dict methods like pop, merge, etc.
    """

    def register(self, name: str, f=None, **kwargs):
        """Register a callable object in this registry.

        Args:
            name: Name of f to register.
            f: A callable object to register. If None,
                this method works as a decorator.
            **kwargs: Default values of f. Useful to register
                callable with different parameters.

        Raises:
            If name is already registered, raise an error.

        Example:
            Directly add a callable object.
            ```python
            f = lambda x,y: x+y
            registry = Registry()

            # new_f is equal to partial(f, y=1).
            new_f = registry.register("add", f, y=1)

            assert registry.get("add")(10) == 11
            assert new_f(10) == 11
            ```

            Use as a decorator.
            ```python
            registry = Registry()

            @registry.register("add", y=1)
            def f(x, y): x+y

            assert registry.get("add")(10) == 11
            ```
        """
        if name in self:
            raise RuntimeError(f"{name} is already registered.")

        if f is None:
            # decorator.
            def deco(f):
                self[name] = functools.partial(f, **kwargs)
                return f

            return deco
        else:
            new_f = functools.partial(f, **kwargs)
            self[name] = new_f
            return new_f


# Create a global registry for the simple usage.
registry = Registry()
register = registry.register
