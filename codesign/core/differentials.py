from codesign.core.scalars import Variable, Atomic
from codesign.core.tree_base import Algebraic, UniqueObject
from codesign.core.vectors import Vector, DenseArray
from codesign.core.vectors import is_scalar
import abc

t = Variable()


class Derivative(Variable, UniqueObject):
    def __init__(self, variable, *args, **kwargs):
        self.parent = variable
        super().__init__(name=f'd{variable.name}')


class VectorDerivative(Vector, UniqueObject):
    def __init__(self, variable):
        data = [Derivative(v) for v in variable.data]
        self.parent = variable
        super().__init__(data=data, name=f'd{variable.name}')


class MatrixDerivative(DenseArray, UniqueObject):
    def __init__(self, variable):
        data = [[Derivative(v) for v in row] for row in variable.data]
        self.parent = variable
        super().__init__(data=data, name=f'd{variable.name}')


__diff_pairing = [
        (Variable, Derivative),
        (Vector, VectorDerivative),
        (DenseArray, MatrixDerivative)
    ]


def diff(variable):
    for base_cls, diff_cls in __diff_pairing:
        if isinstance(variable, base_cls):
            return diff_cls(variable)

    raise TypeError(f"Can't differentiate type {type(variable)}")


def is_differential(variable):
    for _, diff_cls in __diff_pairing:
        if isinstance(variable, diff_cls):
            return True

    return False
