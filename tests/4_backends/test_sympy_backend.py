# Goal:
# - specify sympy as the backend instead of casadi
# - set up a symbolic 2nd order ODE
# - produce symbolic representation
import pytest
import sympy
import sympy as sp
from sysopt.backends import get_backend
from sysopt.symbolic import *
from sysopt.problems.solver import SolverContext

backend = get_backend('sympy')


@pytest.mark.xfail
def test_simple_expression_graph():

    x = Variable('x', 2)
    y = x[0] * x[1] + np.cos(x[1])

    y_s = backend.get_implementation(y)

    assert isinstance(y_s, sympy.Mul)


def test_simple_model(linear_model):

    with SolverContext(model=linear_model,
                       t_final=1,
                       parameters={},
                       backend='sympy') as solver:
        p = sp.symbols(
            ','.join(param.name for param in solver.parameters)
        )
        integrator = solver.get_symbolic_integrator()

        soln = integrator([p])

        assert str(soln['f'][0]) == 'u_0 + x_1'
        assert str(soln['f'][1]) == '-x_0'
