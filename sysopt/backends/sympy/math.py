"""Math operations for the sympy backend"""

# pylint: disable=unused-import
# pylint: disable=invalid-name
import numpy as np
from sympy import (
    sin, cos, tan, asin, acos,  atan2, log, exp, atan,
    sinh, cosh, tanh, asinh, acosh, atanh, Heaviside, DiracDelta,
    sign
)
from sysopt.backends import get_backend
backend = get_backend('sympy')

fabs = abs
heaviside = Heaviside
dirac = DiracDelta


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
    np.power: lambda x, y: x ** y
}

for ufunc, f in ufuncs.items():
    backend.implements(ufunc)(f)
