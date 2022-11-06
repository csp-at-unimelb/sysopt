"""Casadi Function Factories."""
from typing import Dict, NewType, Callable, Any
from importlib import import_module, invalidate_caches

from sysopt.symbolic.core import Algebraic

Factory = NewType('Factory', Callable[[Any], Algebraic])

__backends = {}


def get_backend(name='casadi'):
    global __backends

    try:
        return __backends[name]
    except KeyError:
        pass

    for cls in BackendContext.__subclasses__():
        if cls.name == name:
            instance = cls()
            __backends[name] = instance
            return instance
    raise NotImplementedError(f'Unknown backend: {name}')


class BackendContext:
    """Loads and keeps track of symbolic backend."""

    name = None

    def __init__(self):
        self.__implementations = {}

    def implements(self, sysopt_cls):
        def wrapper(func):
            self.__implementations[str(sysopt_cls)] = func
            return func
        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_implementation(self, obj):
        cls = str(obj.__class__)

        try:
            factory = self.__implementations[cls]
        except KeyError as ex:
            msg = f'Backend doesn\'t know how to turn and object of {cls}' \
                  'into a function'
            raise NotImplementedError(msg) from ex

        return factory(obj)
