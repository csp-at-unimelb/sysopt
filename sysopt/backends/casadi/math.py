"""Casadi Math Functions."""
# pylint: disable=unused-import

from casadi import (
    sin, sinh, asin, asinh,
    cos, cosh, acos, acosh,
    tan, tanh, atan, atanh, atan2,
    exp, log, sign,
    power, fmax, fmin, fabs
)
import casadi as __casadi
import numpy as np
from sysopt.backends.implementation_hooks import get_backend
backend = get_backend('casadi')


def heaviside(x, eps=1e-4):
    return 1/(1 + exp(-2*x/eps))


def dirac(x, eps=1e-4):
    """Dirac delta function"""
    return heaviside(x - eps) + heaviside(eps - x)


def sum_axis(matrix, axis=0):
    if axis == 0:
        return __casadi.sum1(matrix)
    elif axis == 1:
        return __casadi.sum2(matrix)
    raise NotImplementedError


ufuncs = {
    np.sin: sin,
    np.cos: cos,
    np.tan: tan,
    np.arcsin: asin,
    np.arccos: acos,
    np.arctan: atan,
    np.arctan2: atan2,
    np.log: log,
    np.exp: exp,
    np.sinh: sinh,
    np.cosh: cosh,
    np.tanh: tanh,
    np.arcsinh: asinh,
    np.arccosh: acosh,
    np.arctanh: atanh,
    np.sign: sign,
    np.power: power
}

for ufunc, f in ufuncs.items():
    backend.implements(ufunc)(f)
