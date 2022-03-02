import warnings
from dataclasses import dataclass
from typing import Iterable, Callable, Optional


@dataclass
class FlattenedSystem:
    X: Optional[Iterable] = None            # Dynamics
    Z: Optional[Iterable] = None            # Coupling Variables
    U: Optional[Iterable] = None            # Inputs
    P: Optional[Iterable] = None            # Parameters
    f: Optional[Callable] = None            # Explicit Dynamics
    g: Optional[Callable] = None            # Outputs
    h: Optional[Callable] = None            # Algebraic Constraints.
    j: Optional[Callable] = None            # Quadratures
    X0: Optional[Iterable] = None           # Initial values

    def __iadd__(self, other):
        assert isinstance(other, FlattenedSystem)
        backend = get_backend()
        self.X = backend.concatenate(self.X, other.X)
        self.Z = backend.concatenate(self.Z, other.Z)
        self.U = backend.concatenate(self.U, other.U)
        self.P = backend.concatenate(self.P, other.P)
        self.f = backend.concatenate(self.f, other.f)
        self.g = backend.concatenate(self.g, other.g)
        self.h = backend.concatenate(self.h, other.h)
        self.j = backend.concatenate(self.j, other.j)
        self.X0 = backend.concatenate(self.X0, other.X0)
        return self

    def __add__(self, other):
        result = FlattenedSystem()
        result += self
        result += other
        return result


__backend = None


class CodesignSolverContext:

    def concatenate(self, *vectors):
        raise NotImplementedError

    def wrap_function(self, function, *args):
        raise NotImplementedError

    def get_or_create_variables(self, block):
        raise NotImplementedError

    def t(self):
        raise NotImplementedError


def get_default_backend():
    global __backend
    if not __backend:
        from sysopt.backends.casadi import CasadiBackend
        __backend = CasadiBackend()
    return __backend


def get_backend():
    global __backend

    if not __backend:
        __backend = get_default_backend()
        warning = "Symbolic backend not specified " \
                  f"- using default {__backend.name}"
        warnings.warn(warning, UserWarning, stacklevel=1)
        return __backend
    return __backend


def is_leaf(block):
    return not hasattr(block, 'components')
