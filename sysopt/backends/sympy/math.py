"""Math operations for the sympy backend"""

# pylint: disable=unused-import
# pylint: disable=invalid-name

from sympy import (
    sin, cos, tan, asin, acos,  atan2, log, exp,
    sinh, cosh, tanh, asinh, acosh, atanh, Heaviside, DiracDelta,
    sign
)

fabs = abs
heaviside = Heaviside
dirac = DiracDelta
