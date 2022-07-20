"""Casadi Function Factories."""
from typing import Dict, NewType, Callable, Any, List
from ordered_set import OrderedSet
import casadi
from sysopt.symbolic.symbols import Algebraic, ConstantFunction, SymbolicArray


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


class CasadiImplementation(Algebraic):
    def __init__(self, func, shape, arguments: List[SymbolicArray]):
        self.func = func
        self._shape = shape
        self.arguments = arguments

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return repr(self.func)

    @property
    def shape(self):
        return self._shape

    def symbols(self):
        return OrderedSet(self.arguments)

    def __call__(self, *args):
        result = self.func(*args)
        try:
            return result.full()
        except AttributeError:
            return result

    def pushforward(self, *args):
        n = len(self.symbols())
        assert len(args) == 2 * n, f'expected {2 * n} arguments, ' \
                                   f'got {len(args)}'
        x, dx = args[:n], args[n:]
        out_sparsity = casadi.DM.ones(*self.shape)
        jac = self.func.jacobian()(*x, out_sparsity)
        dx = casadi.vertcat(*dx)
        result = jac @ dx

        return self.func(*x), result


@implements(ConstantFunction)
def to_constant(func: ConstantFunction):
    return lambda x: casadi.SX(func.value)


