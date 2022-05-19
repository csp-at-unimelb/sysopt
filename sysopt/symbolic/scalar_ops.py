"""Symbolic scalar/component-wise operations"""
import numpy as np
from sysopt.symbolic.symbols import wrap_as_op, implements, numpy_handlers

import sysopt.backends as backend

exp = wrap_as_op(backend.exp, 1, numpy_func=np.exp)

log = wrap_as_op(backend.log, 1, numpy_func=np.log)
sin = wrap_as_op(backend.sin, 1, numpy_func=np.sin)
cos = wrap_as_op(backend.cos, 1, numpy_func=np.cos)
tan = wrap_as_op(backend.tan, 1, numpy_func=np.tan)
asin = wrap_as_op(backend.asin, 1, numpy_func=np.arcsin)
acos = wrap_as_op(backend.acos, 1, numpy_func=np.arccos)
atan = wrap_as_op(backend.atan, 1, numpy_func=np.arctan)
sinh = wrap_as_op(backend.sinh, 1, numpy_func=np.sinh)
cosh = wrap_as_op(backend.cosh, 1, numpy_func=np.cosh)
tanh = wrap_as_op(backend.tanh, 1, numpy_func=np.tanh)
asinh = wrap_as_op(backend.asinh, 1, numpy_func=np.arcsinh)
acosh = wrap_as_op(backend.acosh, 1, numpy_func=np.arccosh)
atanh = wrap_as_op(backend.atanh, 1, numpy_func=np.arctanh)

heaviside = wrap_as_op(backend.heaviside, 1)
numpy_handlers[np.heaviside] = lambda x, x0, *args, **kwargs: heaviside(x)

dirac = wrap_as_op(backend.dirac, 1)

sign = wrap_as_op(backend.sign, 1, numpy_func=np.sign)
atan2 = wrap_as_op(backend.atan2, 2, numpy_func=np.arctan2)

unary = [
    sin, cos, asin, acos, tan, atan,
    sinh, asinh, cosh, acosh, tanh, atanh,
    exp, log, heaviside, dirac, sign
]

binary = [
    atan2   # add, subtract, multiply, divide
]
