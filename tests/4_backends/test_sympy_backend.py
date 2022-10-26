# Goal:
# - specify sympy as the backend instead of casadi
# - set up a symbolic 2nd order ODE
# - produce symbolic representation
import pytest
import sympy
import sympy as sp
from sysopt.backends import BackendContext
from sysopt.symbolic import *
from sysopt.problems.solver import SolverContext

def test_sympy_is_backend():
    with BackendContext('sympy') as be:
        x = sp.symbols('x')
        assert be.cos(x) == sp.cos(x)


def test_simple_expression_graph():

    x = Variable('x', 2)
    y = x[0] * x[1]

    with BackendContext('sympy') as backend:
        y_s = backend.get_implementation(y)

    assert isinstance(y_s, sympy.Mul)


def test_simple_model(linear_model):

    with SolverContext(model=linear_model,
                       t_final=1,
                       constants={},
                       backend='sympy') as solver:
        print(solver.parameters)
        soln = solver.get_symbolic_integrator()


    assert False
