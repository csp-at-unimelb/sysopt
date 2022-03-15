from sysopt.symbolic import concatenate
from typing import Iterable, Callable, Optional, Tuple, Dict
from dataclasses import dataclass, field
from sysopt.block import Block

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
        self.X = concatenate(self.X, other.X)
        self.Z = concatenate(self.Z, other.Z)
        self.U = concatenate(self.U, other.U)
        self.P = concatenate(self.P, other.P)
        self.f = concatenate(self.f, other.f)
        self.g = concatenate(self.g, other.g)
        self.h = concatenate(self.h, other.h)
        self.j = concatenate(self.j, other.j)
        self.X0 = concatenate(self.X0, other.X0)


        return self

    def __add__(self, other):
        result = FlattenedSystem()
        result += self
        result += other
        return result
