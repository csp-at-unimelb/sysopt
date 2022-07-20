from sysopt.symbolic import Function, symbolic_vector
from sysopt.backends import get_implementation
from sysopt.backends.casadi.foreign_function import CasadiFFI
from casadi import SX, MX


def f(x, y):
    return x + y, x**2 + y**2


def dfdx(x, y):
    return (1, 1), (2 * x, 2*y)


class TestForeignFunction:
    def test_casadi_ffi(self):
        x = symbolic_vector('X',1)
        y = symbolic_vector('y', 1)
        callback = CasadiFFI(f, [x, y], (2, 1))

        result = callback(1, 2)
        assert result[0] == 3
        assert result[1] == 5


    def test_compile_function(self):
        f_spec = Function(
            function=f,
            jacobians=dfdx,
            shape=(2,),
            arguments=[symbolic_vector('x'), symbolic_vector('y')]
        )
        F = get_implementation(f_spec)

        result = F(1, 2)

        assert result[0] == 3
        assert result[1] == 5
