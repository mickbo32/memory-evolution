# from abc import ABC, abstractmethod, ABCMeta  # see for inspiration: lib/python3.9/_py_abc.py -> ABCMeta
# from functools import wraps


def override(funcobj):
    """Override decorator.

    Requires that the metaclass is MustOverrideMeta or derived from
    a class which the metaclass is MustOverrideMeta (e.g. MustOverride).
    If a class method use it, and the class is later subclassed,
    the subclass needs override the method, otherwise a TypeError
    will be raised."""
    funcobj.__override__ = True
    return funcobj


class NotOverriddenError(TypeError):
    """Raised when an attribute that must be overridden is not."""

    def __init__(self, attrs, *args, msg=None, **kwargs):
        self.attrs = attrs
        self.msg = msg

    def __str__(self):
        # msg = self.msg
        # if msg is None:
        #     msg = f"Some attributes which must be overridden are not overridden: {self.attrs}"
        # return msg
        return self.msg if self.msg is not None else (
            f"Some attributes which must be overridden are not overridden: {self.attrs}"
        )


class MustOverrideMeta(type):

    def __new__(mcs, name, bases, dct, **kwargs):
        cls = super().__new__(mcs, name, bases, dct, **kwargs)
        override_attrs = {attr
                          for attr, value in dct.items()
                          if getattr(value, "__override__", False)}
        cls.__override_attrs__ = frozenset(override_attrs)
        not_overridden_attrs = set()
        for base in bases:
            for name in getattr(base, "__override_attrs__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__override__", False) and name not in override_attrs:
                    not_overridden_attrs.add(name)
        if not_overridden_attrs:
            raise NotOverriddenError(not_overridden_attrs)
        return cls


class MustOverride(metaclass=MustOverrideMeta):
    pass

