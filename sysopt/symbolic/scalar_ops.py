"""Symbolic scalar/component-wise operations"""
from sysopt.symbolic.symbols import wrap_as_op

import sysopt.backends as backend

exp = wrap_as_op(backend.exp, 1)
log = wrap_as_op(backend.log, 1)
sin = wrap_as_op(backend.sin, 1)
cos = wrap_as_op(backend.cos, 1)
tan = wrap_as_op(backend.tan, 1)
asin = wrap_as_op(backend.asin, 1)
acos = wrap_as_op(backend.acos, 1)
atan = wrap_as_op(backend.atan, 1)
sinh = wrap_as_op(backend.sinh, 1)
cosh = wrap_as_op(backend.cosh, 1)
tanh = wrap_as_op(backend.tanh, 1)
asinh = wrap_as_op(backend.asinh, 1)
acosh = wrap_as_op(backend.acosh, 1)
atanh = wrap_as_op(backend.atanh, 1)

heaviside = wrap_as_op(backend.heaviside, 1)
dirac = wrap_as_op(backend.dirac, 1)
sign = wrap_as_op(backend.sign, 1)
atan2 = wrap_as_op(backend.atan2, 2)

unary = [
    sin, cos, asin, acos, tan, atan,
    sinh, asinh, cosh, acosh, tanh, atanh,
    exp, log, heaviside, dirac, sign
]

binary = [
    atan2   # add, subtract, multiply, divide
]
