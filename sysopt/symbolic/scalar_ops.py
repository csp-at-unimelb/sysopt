"""Symbolic scalar/component-wise operations"""
import numpy as np
from sysopt.symbolic.core import wrap_as_op, numpy_handlers
from sysopt.backends import get_backend


exp = wrap_as_op(lambda x: get_backend().exp(x), 1, numpy_func=np.exp)
log = wrap_as_op(lambda x: get_backend().log(x), 1, numpy_func=np.log)
sin = wrap_as_op(lambda x: get_backend().sin(x), 1, numpy_func=np.sin)
cos = wrap_as_op(lambda x: get_backend().cos(x), 1, numpy_func=np.cos)
tan = wrap_as_op(lambda x: get_backend().tan(x), 1, numpy_func=np.tan)
asin = wrap_as_op(lambda x: get_backend().asin(x), 1, numpy_func=np.arcsin)
acos = wrap_as_op(lambda x: get_backend().acos(x), 1, numpy_func=np.arccos)
atan = wrap_as_op(lambda x: get_backend().atan(x), 1, numpy_func=np.arctan)
sinh = wrap_as_op(lambda x: get_backend().sinh(x), 1, numpy_func=np.sinh)
cosh = wrap_as_op(lambda x: get_backend().cosh(x), 1, numpy_func=np.cosh)
tanh = wrap_as_op(lambda x: get_backend().tanh(x), 1, numpy_func=np.tanh)
asinh = wrap_as_op(lambda x: get_backend().asinh(x), 1, numpy_func=np.arcsinh)
acosh = wrap_as_op(lambda x: get_backend().acosh(x), 1, numpy_func=np.arccosh)
atanh = wrap_as_op(lambda x: get_backend().atanh(x), 1, numpy_func=np.arctanh)
fabs = wrap_as_op(lambda x: get_backend().fabs(x), 1, numpy_func=np.abs)

heaviside = wrap_as_op(lambda x: get_backend().heaviside(x), 1)
numpy_handlers[np.heaviside] = lambda x, x0, *args, **kwargs: heaviside(x)

dirac = wrap_as_op(lambda x: get_backend().dirac(x), 1)

sign = wrap_as_op(lambda x: get_backend().sign(x), 1, numpy_func=np.sign)
atan2 = wrap_as_op(lambda y, x: get_backend().atan2(y, x), 2, numpy_func=np.arctan2)

unary = [
    sin, cos, asin, acos, tan, atan,
    sinh, asinh, cosh, acosh, tanh, atanh,
    exp, log, heaviside, dirac, sign
]

binary = [
    atan2   # add, subtract, multiply, divide
]
