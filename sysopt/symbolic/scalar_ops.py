"""Symbolic scalar/component-wise operations"""
from sysopt.symbolic.symbols import wrap_function

import sysopt.backends as backend

exp = wrap_function(backend.exp, 1)
log = wrap_function(backend.log, 1)
sin = wrap_function(backend.sin, 1)
cos = wrap_function(backend.cos, 1)
tan = wrap_function(backend.tan, 1)
asin = wrap_function(backend.asin, 1)
acos = wrap_function(backend.acos, 1)
atan = wrap_function(backend.atan, 1)
sinh = wrap_function(backend.sinh, 1)
cosh = wrap_function(backend.cosh, 1)
tanh = wrap_function(backend.tanh, 1)
asinh = wrap_function(backend.asinh, 1)
acosh = wrap_function(backend.acosh, 1)
atanh = wrap_function(backend.atanh, 1)

heaviside = wrap_function(backend.heaviside, 1)
dirac = wrap_function(backend.dirac, 1)
sign = wrap_function(backend.sign, 1)
atan2 = wrap_function(backend.atan2, 2)

unary = [
    sin, cos, asin, acos, tan, atan,
    sinh, asinh, cosh, acosh, tanh, atanh,
    exp, log, heaviside, dirac, sign
]

binary = [
    atan2   # add, subtract, multiply, divide
]
