"""Casadi Function Factories."""
from typing import Dict, NewType, Callable, Any
import casadi
from sysopt.symbolic.core import Algebraic, ConstantFunction


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
        msg = f'Casadi backend doesn\'t know how to turn and object of {cls}' \
              'into a function'
        raise NotImplementedError(msg) from ex

    return factory(obj)


@implements(ConstantFunction)
def to_constant(func: ConstantFunction):
    return lambda x: casadi.MX(func.value)


