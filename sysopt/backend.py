"""API definitions for symbolic backends."""
# pylint: disable=invalid-name

from abc import ABCMeta
import warnings
from dataclasses import dataclass
from typing import Iterable, Callable, Optional


@dataclass
class FlattenedSystem:
    """Intermediate representation of a systems model."""
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


class ADContext(metaclass=ABCMeta):
    """Interface for automatic differentiation and solver backends."""
    def concatenate(self, *vectors):
        raise NotImplementedError

    def wrap_function(self, function, *args):
        raise NotImplementedError

    def get_or_create_variables(self, block):
        raise NotImplementedError

    def get_or_create_outputs(self, block):
        raise NotImplementedError

    @property
    def t(self):
        raise NotImplementedError

    def get_or_create_port_variables(self, port):
        if port is port.parent.inputs:
            _, _, u, _ = self.get_or_create_variables(port.parent)
            return u
        if port is port.parent.outputs:
            return self.get_or_create_outputs(port.parent)

    def sparse_matrix(self, shape):
        raise NotImplementedError

    def get_or_create_decision_variable(self, block=None, parameters=None):
        raise NotImplementedError


def get_default_backend():
    """Return the default solver backend (Casadi) """
    # pylint: disable=import-outside-toplevel
    global __backend
    if not __backend:
        from sysopt.backends.casadi import CasadiBackend
        __backend = CasadiBackend()
    return __backend


def get_backend():
    """Return the current solver backend."""
    global __backend

    if not __backend:
        __backend = get_default_backend()
        warning = 'Symbolic backend not specified ' \
                  f'- using default {__backend.name}'
        warnings.warn(warning, UserWarning, stacklevel=1)
        return __backend
    return __backend


def projection_matrix(indices, dimension):

    matrix = __backend.sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


def signal(parent, indices, t):
    backend = get_backend()

    vector = backend.get_or_create_port_variables(parent)
    matrix = projection_matrix(list(enumerate(indices)), len(vector))

    # get parent signal
    # construct a projection matrix onto the indices
    # compose with evaluator
    #
    # dirac(t) * proj(indicis) @ parent.signal(t)

    return matrix @ vector
