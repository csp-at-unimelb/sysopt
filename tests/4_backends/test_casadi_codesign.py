import casadi
import sympy as sp
from sysopt.modelling.types import *
from sysopt.modelling.block import Block
from sysopt.problems.solver import CasadiContext
from sysopt.modelling.block import Composite
from sysopt.modelling.builders import FullStateOutput
from sysopt.blocks import ConstantSignal
from sysopt.symbolic import PiecewiseConstantSignal
from sysopt.backends.casadi.codesign_solver import (
    CasadiCodesignProblemData, transcribe_problem, CodesignSolverOptions
)
from sysopt import Domain


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

    with CasadiContext(model=block, t_final=1) as solver:
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


class TestCasadiCodesign:
    def test_constrained_functional(self):
        # vector field
        # dq = -p
        # dp = q + u
        # u = p
        t = casadi.MX.sym('t')
        x_0 = casadi.MX.sym('x_0')
        x_1 = casadi.MX.sym('x_1')
        u = casadi.MX.sym('u')
        p = casadi.MX.sym('p')
        q = casadi.MX.sym('q')
        y = casadi.MX.sym('y', 3)
        x = casadi.vertcat(x_0, x_1)
        args = [t, x, u, p]
        vector_field = casadi.Function('f', args, [10 * casadi.vertcat(-x_1 + u, x_0)])
        outputs = casadi.Function('g', args, [casadi.vertcat(x_0, x_1, u)])
        constraints = casadi.Function('h', args, [u - p])
        quadrature = casadi.Function('q_dot', [t, y, p], [10 * y[2]])
        intitial_conditions = casadi.Function('x0', [p], [casadi.MX([0, 1])])
        cost = casadi.Function('cost', [t, y, q, p], [y[0]**2 + y[1]**2])
        parameters = {
            PiecewiseConstantSignal('u', shape=(1,), frequency=10): (-1, 1)
        }
        path_constraint = casadi.Function(
            'c_t', [t, y, q, p], [casadi.MX()]
        )
        terminal_constraint = casadi.Function(
            'c_T', [t, y, q, p], [casadi.MX()]
        )
        t_final = casadi.Function('t_f', [p], [10])
        data = CasadiCodesignProblemData(
            domain=Domain(1, 2, 0, 1, 1),
            vector_field=vector_field,
            outputs=outputs,
            algebraic_constraint=constraints,
            quadrature=quadrature,
            initial_conditions=intitial_conditions,
            cost_function=cost,
            parameters=parameters,
            path_constraints=path_constraint,
            terminal_constraints=terminal_constraint,
            final_time=t_final
        )

        options = CodesignSolverOptions(
            grid_size=100,
            degree=4
        )
        solver, solver_args, argmin, path = transcribe_problem(
            data, options, [0]
        )
        result = solver(**solver_args)
        soln = result['x']
        control = argmin(soln)
        t, y, q,_,_ = path(soln)
        y = y.full()
        assert np.linalg.norm(y[0:2, -1]) < 1e-3

    def test_codesign_problem_with_path_variable(self):
        model = Composite(name='Test Model')
        # build a LQR model
        #
        plant_metadata = Metadata(
            inputs=['u'],
            states=['x_0', 'x_1']
        )
        A = np.array([[0, 1],
                      [-1, 0]], dtype=float)
        B = np.array([[1], [0]], dtype=float)

        def f(t, x, u, _):
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
        t_final = 1
        with CasadiContext(model, t_final=t_final) as context:
            y_t = context.integrate([0])
            y_f = y_t.x[0:2, -1]
            cost_f = y_f.T @ y_f
            y = model.outputs(t_final)
            constraint = [
                u <= 1,
                u >= -1
            ]
            problem = context.problem(
                [u],
                cost=y[0]**2 + y[1]**2,
                subject_to=constraint
            )
            sol = problem.solve([0])

            assert sol.cost < cost_f

