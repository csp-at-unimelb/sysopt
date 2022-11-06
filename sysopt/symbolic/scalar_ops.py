"""Symbolic scalar/component-wise operations"""
import numpy as np
from sysopt.symbolic.core import wrap_ufunc

exp = wrap_ufunc(np.exp, 1)
log = wrap_ufunc(np.log, 1)
sin = wrap_ufunc(np.sin, 1)
cos = wrap_ufunc(np.cos, 1)
tan = wrap_ufunc(np.tan, 1)
asin = wrap_ufunc(np.arcsin, 1)
acos = wrap_ufunc(np.arccos, 1)
atan = wrap_ufunc(np.arctan, 1)
sinh = wrap_ufunc(np.sinh, 1)
cosh = wrap_ufunc(np.cosh, 1)
tanh = wrap_ufunc(np.tanh, 1)
asinh = wrap_ufunc(np.arcsinh, 1)
acosh = wrap_ufunc(np.arccosh, 1)
atanh = wrap_ufunc(np.arctanh, 1)
fabs = wrap_ufunc(np.abs, 1)


heaviside = wrap_ufunc(np.heaviside, 1)

sign = wrap_ufunc(np.sign, 1)
atan2 = wrap_ufunc(np.arctan2, 2)

unary = [
    sin, cos, asin, acos, tan, atan,
    sinh, asinh, cosh, acosh, tanh, atanh,
    exp, log, heaviside, sign
]

binary = [
    atan2   # add, subtract, multiply, divide
]
