"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Iterable, Callable, Optional
from dataclasses import dataclass


from sysopt.backends import (
    SymbolicVector, concatenate_symbols, lambdify
)
from sysopt.symbolic import Variable
from sysopt.blocks.block_operations import create_functions_from_block
from sysopt.helpers import flatten


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
    t = SymbolicVector('t', 1)

    def __iadd__(self, other):
        assert isinstance(other, FlattenedSystem)
        self.X = concatenate_symbols(self.X, other.X)
        self.Z = concatenate_symbols(self.Z, other.Z)
        self.U = concatenate_symbols(self.U, other.U)
        self.P = concatenate_symbols(self.P, other.P)
        self.f = concatenate_symbols(self.f, other.f)
        self.g = concatenate_symbols(self.g, other.g)
        self.h = concatenate_symbols(self.h, other.h)
        self.j = concatenate_symbols(self.j, other.j)
        self.X0 = concatenate_symbols(self.X0, other.X0)

        return self

    def lambdify(self):
        args = self.arguments
        f = lambdify(self.f, args, 'f') if self.f is not None else None
        g = lambdify(self.g, args, 'g') if self.g is not None else None
        h = lambdify(self.h, args, 'h') if self.h is not None else None
        j = lambdify(self.j, args, 'j') if self.j is not None else None
        x0 = lambdify(self.X0, self.P, 'x0') if self.X0 is not None else None
        return f, g, h, j, x0

    @property
    def arguments(self):
        args = [
            v for v in (self.t, self.X, self.Z, self.U, self.P)
            if v is not None
        ]

        return args

    def __add__(self, other):
        result = FlattenedSystem()
        result += self
        result += other
        return result

    def add_quadrature(self, function):
        try:
            idx = len(self.j)
            self.j = concatenate_symbols(self.j, function)
        except TypeError:
            idx = 0
            self.j = function
        return idx

    @staticmethod
    def from_block(block):
        x0, f, g, h, _ = create_functions_from_block(block)
        domain = g.domain
        self = FlattenedSystem()
        self.X = SymbolicVector('x', domain.states)
        self.Z = SymbolicVector('z', domain.constraints)
        self.U = SymbolicVector('u', domain.inputs)
        self.P = SymbolicVector('p', domain.parameters)
        args = (self.t,
                self.X.tolist(),
                self.Z.tolist(),
                self.U.tolist(),
                self.P.tolist())

        if f:
            self.f = f(*args)
            self.X0 = x0(self.P.tolist())
            assert f.domain == domain
        if g:
            assert g.domain == domain
            r = flatten(g(*args))
            self.g = r
        if h:
            assert h.domain == domain
            self.h = h(*args)
        return self


class SymbolDatabase:
    """Autodiff context"""

    def __init__(self, t_final=1):
        self.t_final = t_final
        self._t = SymbolicVector('t', 1)
        self._free_variables = {}

        if isinstance(t_final, Variable):
            self._free_variables['T'] = t_final

    @property
    def t(self):
        return self._t
