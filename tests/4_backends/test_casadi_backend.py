import casadi
import numpy as np
from sysopt.symbolic import Function, symbolic_vector
from sysopt.backends import get_implementation
from sysopt.backends.casadi.foreign_function import CasadiFFI
from casadi import MX


def f(x, y):
    return x + y, x**2 + y**2


def dfdx(x, y):
    return (1, 1), (2 * x, 2*y)


class TestForeignFunction:
    def test_compile_function(self):
        f_spec = Function(
            function=f,
            jacobian=dfdx,
            shape=(2,),
            arguments=[symbolic_vector('x'), symbolic_vector('y')]
        )
        F = get_implementation(f_spec)

        result = F(1, 2)

        assert result[0] == 3
        assert result[1] == 5

    def test_casadi_ffi(self):
        x = symbolic_vector('X',1)
        y = symbolic_vector('y', 1)
        callback = CasadiFFI(f, [x, y], (2, 1))

        result = callback([1, 2])
        assert result[0] == 3
        assert result[1] == 5

    def test_jacobian(self):
        f_spec = Function(
            function=f,
            jacobian=dfdx,
            shape=(2,),
            arguments=[symbolic_vector('x'), symbolic_vector('y')]
        )
        F = get_implementation(f_spec).impl

        x = MX.sym('x', 2)
        J = casadi.Function('J', [x], [casadi.jacobian(F(x), x)])
        result = J([1, 2])
        assert result[0, 0] == 1  # df_0/dx_0
        assert result[0, 1] == 1  # df_0/dx_1
        assert result[1, 0] == 2  # df_1/dx_0
        assert result[1, 1] == 4  # df_1/dx_1