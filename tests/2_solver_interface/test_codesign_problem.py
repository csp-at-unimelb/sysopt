import numpy as np
import sympy as sp
from sysopt.types import *
from sysopt.block import Block
from sysopt.solver.solver import Problem, SolverContext, get_time_variable
from sysopt.block import Composite
from sysopt.blocks import FullStateOutput, ConstantSignal
from sysopt.symbolic import Variable, PiecewiseConstantSignal

class LinearScalarEquation(Block):
    r"""Linear ODE of the form

    math::
        \dot{x} = -ax
        x(0)  = x_0

    """

    def __init__(self):
        metadata = Metadata(
            inputs=[],
            outputs=['y'],
            states=['x'],
            constraints=[],
            parameters=['a', 'x0']
        )
        super().__init__(metadata, 'dx')

    def initial_state(self, parameters: Parameters) -> Numeric:
        _, x0 = parameters
        return x0

    def compute_dynamics(self,
                         t: Time,
                         states: States,
                         algebraics: Algebraics,
                         inputs: Inputs,
                         parameters: Parameters):
        x, = states
        a, _ = parameters
        return -x*a

    def compute_outputs(self,
                        t: Time,
                        states: States,
                        algebraics: Algebraics,
                        inputs: Inputs,
                        parameters: Parameters) -> Numeric:
        x, = states
        return x

    def explicit_solution(self, t, parameters):
        a, x0 = parameters
        return np.exp(-a * t) * x0

    def dxdp(self, t, parameters):
        a, x0 = parameters
        return [
            -t * a * np.exp(-a*t) * x0,
            np.exp(-a * t)
        ]

    def pushforward(self, t,  p, dp):
        x = self.explicit_solution(t, p)
        a, x0 = p
        return [
            -t * a * x * dp[0] + x * dp[1] / x0
        ]


class Problem1:
    @staticmethod
    def symbolic_cost():
        t, a, x0 = sp.symbols('t ,a, x0')
        x = x0 * sp.exp(-a * t)
        j = Problem1.running_cost(x)
        cost = sp.integrate(j, (t, 0, t)) - x
        return cost, a, x0, t

    @staticmethod
    def running_cost(y):
        return - y ** 2

    @staticmethod
    def cost(t, params):
        a, x0 = params
        cost, *symbols = Problem1.symbolic_cost()
        subs = dict(zip(symbols, [a,x0, t]))
        result = float(cost.evalf(subs=subs))
        return result
        # return x0**2 *(np.exp(-2*a * t) - 1) / (2 * a)

    @staticmethod
    def tangent_space(params, t):
        cost, sa, sx0, st = Problem1.symbolic_cost()
        a, x0 = params
        subs = {sa: a, sx0: x0, st: t}
        dfda = sp.diff(cost, sa).evalf(subs=subs)
        dfdx0 = sp.diff(cost, sx0).evalf(subs=subs)
        return np.array([[dfda], [dfdx0]])


def test_codesign_problem_1():
    block = LinearScalarEquation()

    with SolverContext(model=block, t_final=1) as solver:
        params = solver.parameters

        assert len(params) == 2

        constraints = [
            0.5 < params[0] < 1,
            0.5 < params[1] < 1
        ]
        t = solver.t
        y = block.outputs(t)

        running_cost = -y ** 2
        p0 = [0.75, 0.75]

        cost = solver.integral(running_cost) - y(1)
        problem = solver.problem(params, cost, constraints)

        result = float(problem(p0)[0])

        assert abs(result - Problem1.cost(1, p0)) < 1e-4

        jac = problem.jacobian(p0)
        assert jac.shape == (2, 1)

        grad_known = Problem1.tangent_space(p0, t=1)
        assert (jac - grad_known < 1e-4).all()


def test_codesign_problem_with_path_variable():
    model = Composite(name='Test Model')
    # build a LQR model
    #
    plant_metadata = Metadata(
        inputs=['u'],
        outputs=['x_0', 'x_1']
    )
    A = np.array([[0, 1],
                  [-1, 0]], dtype=float)
    B = np.array([])

    def f(x, u, _):
        return A @ x + B @ u

    def x0(_):
        return np.array([0, 1])

    plant = FullStateOutput(
        dxdt=f,
        metadata=plant_metadata,
        x0=x0,
        name='plant'
    )
    control = ConstantSignal(['u'])
    model.components = [plant, control]
    model.declare_outputs(['x_0', 'x_1', 'u'])
    model.wires = [
        (control.outputs, plant.inputs),
        (control.outputs[0], model.outputs[2]),
        (plant.outputs[0], model.outputs[0]),
        (plant.outputs[1], model.outputs[1])
    ]

    u = PiecewiseConstantSignal('u', frequency=10)
    t_final = Variable('t_f')
    with SolverContext(model, t_final=t_final) as context:
        y = model.outputs(t_final)
        constraint = [
            y[0:2].T @ y[0:2] < 1e-9
        ]
        problem = context.problem(
            [t_final, u],
            cost=t_final,
            subject_to=constraint
        )
