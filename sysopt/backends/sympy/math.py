"""sympy Math Functions."""
# pylint: disable=unused-import

from sympy import (
    sin, sinh, asin, asinh,
    cos, cosh, acos, acosh,
    tan, tanh, atan, atanh, atan2,
    exp, log, 
    Pow as power,
    Max as fmax, Min as fmin, sign
)

def heaviside(x, eps=1e-4):
    return 1/(1 + exp(-2*x/eps))


def dirac(x, eps=1e-4):
    """Dirac delta function"""
    return heaviside(x - eps) + heaviside(eps - x)
