"""Casadi Math Functions."""
# pylint: disable=unused-import

from casadi import (
    sin, sinh, asin, asinh,
    cos, cosh, acos, acosh,
    tan, tanh, atan, atanh, atan2,
    exp, log, power,
    fmax, fmin, sign,
    heaviside
)
import casadi as __casadi


def dirac(x, eps=0):
    """Dirac delta function"""
    return heaviside(x - eps) + heaviside(eps - x)


def sum_axis(matrix, axis=0):
    if axis == 0:
        return __casadi.sum1(matrix)
    elif axis == 1:
        return __casadi.sum2(matrix)
    raise NotImplementedError
