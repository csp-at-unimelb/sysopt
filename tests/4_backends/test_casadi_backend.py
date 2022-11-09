import casadi
import numpy as np
from sysopt.symbolic import Function, symbolic_vector
from sysopt.backends import get_backend
from sysopt.backends.casadi.foreign_function import CasadiFFI
from casadi import MX

backend = get_backend('casadi')

def f(x, y):
    return x + y, x**2 + y**2


def dfdx(x, y):
    d_dx = (1, 2*x)
    d_dy = (1, 2*y)
    return d_dx,d_dy


class TestForeignFunction:
    def test_compile_function(self):
        f_spec = Function(
            function=f,
            jacobian=dfdx,
            shape=(2,),
            arguments=[symbolic_vector('x'), symbolic_vector('y')]
        )
        F = backend.get_implementation(f_spec)

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
        F = backend.get_implementation(f_spec).impl

        x = MX.sym('x', 2)
        J = casadi.Function('J', [x], [casadi.jacobian(F(x), x)])
        result = J([1, 2]).full()
        expected_result = np.array(dfdx(1, 2)).T
        assert (np.abs(result - expected_result) < 1e-4).all()

    def test_nonsquare_jacobian(self):
        def f_3(x, y, z):
            return x + y + x* y *z, z**2

        def jac_f_3(x, y, z):
            dfdx = np.zeros(shape=(2, 3), dtype=float)
            dfdx[0, 0] = 1
            dfdx[0, 1] = 1
            dfdx[1, 2] = z

            return dfdx[:, 0:1], dfdx[:, 1:2], dfdx[:, 2: 3]

        f_spec = Function(
            function=f_3,
            jacobian=jac_f_3,
            arguments=[symbolic_vector('x'), symbolic_vector('y'), symbolic_vector('z')],
            shape=(2,)
        )
        F = backend.get_implementation(f_spec)

        x = MX.sym('x', 3)
        F_sym = F(x[0], x[1], x[2])
        jac_F = casadi.Function('J', [x], [casadi.jacobian(F_sym, x)])
        result = jac_F([1, 2, 3])
        expected_result = np.hstack(jac_f_3(1, 2, 3))

        assert (np.abs(result - expected_result) < 1e-4).all()


    def test_forwards_derivative(self):
        def f_3(x, y, z):
            return x + y, z**2

        def df_3(x, y, z, dx, dy, dz):
            dfdx = np.zeros(shape=(2, 3), dtype=float)
            dfdx[0, 0] = 1
            dfdx[0, 1] = 1
            dfdx[1, 2] = z
            dX = np.array([dx, dy, dz]).reshape((3, 1))
            return dfdx @ dX

        f_spec = Function(
            function=f_3,
            forwards=df_3,
            arguments=[symbolic_vector('x'), symbolic_vector('y'), symbolic_vector('z')],
            shape=(2,)
        )
        assert f_spec.forwards is not None

        def jac_f_3(x, y, z):
            dfdx = np.zeros(shape=(2, 3), dtype=float)
            dfdx[0, 0] = 1
            dfdx[0, 1] = 1
            dfdx[1, 2] = z

            return dfdx[:, 0:1], dfdx[:, 1:2], dfdx[:, 2: 3]

        F = backend.get_implementation(f_spec)

        x = MX.sym('x', 3)
        F_sym = F(x[0], x[1], x[2])
        jac_F = casadi.Function('J', [x], [casadi.jacobian(F_sym, x)])
        result = jac_F([1, 2, 3])
        test_result = np.hstack(
            [df_3(1, 2, 3, 1, 0, 0),
             df_3(1, 2, 3,  0, 1, 0),
             df_3(1, 2, 3, 0,  0, 1)])

        expected_result = np.hstack(jac_f_3(1, 2, 3))
        assert (np.abs(test_result - expected_result) < 1e-4).all()

        assert (np.abs(result - expected_result) < 1e-4).all()


