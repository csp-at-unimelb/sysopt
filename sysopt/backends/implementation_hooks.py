"""Casadi Function Factories."""
from typing import Dict, NewType, Callable, Any
from sysopt.symbolic.core import Algebraic

Factory = NewType('Factory', Callable[[Any], Algebraic])


__function_factories: Dict[str, Factory] = {}


def implements(cls):
    """Decorator that registers the decorated function as the 'compiler'

    Args:
        cls: Class

    """
    def wrapper(func):
        __function_factories[str(cls)] = func
        return func
    return wrapper


def get_implementation(obj):
    cls = str(obj.__class__)

    try:
        factory = __function_factories[cls]
    except KeyError as ex:
        msg = f'Backend doesn\'t know how to turn and object of {cls}' \
              'into a function'
        raise NotImplementedError(msg) from ex

    return factory(obj)




