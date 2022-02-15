from codesign import Variable, Vector, Parameter
from codesign.backends.pure_python import lambdify, evaluate, evaluate_atom
import numpy as np


def test_lambdify_scalar():
    p1 = Parameter()
    v1 = Variable()

    expression = p1 == v1

    residue = lambdify(expression, p1, v1)
    result = residue(1, 1)
    assert result == 0

    result = residue(1, 0)

    assert abs(result) == 1


def test_lambdify_vector():
    p = Vector.create_filled_with(Parameter, 3)
    v = Vector.create_filled_with(Variable, 3)

    expression = p == v

    residue = lambdify(expression, p, v)
    ones = np.array([1.0, 1.0, 1.0])
    zeros = np.zeros_like(ones)

    result = residue(ones, ones)

    assert isinstance(result, np.ndarray)
    assert (result == 0).all()

    result = residue(ones, zeros)
    assert (abs(result) == 1).all()
