# Symbolic integration
#
# Given a vector field $\Phi(t, x, v, z, p) = 0
# and a observation function y(t, x(t; p), z(t; p), p)
# where x(0), v(0), z(0) = f_0(p)
# define Y(t, p) := y(t , x(t, p), z(t, p), p)

# want to compute
# D_{t,p}Y  <- jacobian
# or D_{t,p}Y @ D_YL

# Requirements:
# - fixed window of integration
import numpy
import numpy as np

from sysopt.types import *
from sysopt.block import Block
from sysopt.solver import SolverContext


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
            state=['x'],
            constraints=[],
            parameters=['a', 'x0']
        )
        super().__init__(metadata, 'dx')

    def initial_state(self, parameters: Parameters) -> Numeric:
        _, x0 = parameters
        return [x0]

    def compute_dynamics(self,
                         t: Time,
                         state: States,
                         algebraics: Algebraics,
                         inputs: Inputs,
                         parameters: Parameters):
        x, = state
        a, _ = parameters
        return [-x*a]

    def compute_outputs(self,
                        t: Time,
                        state: States,
                        algebraics: Algebraics,
                        inputs: Inputs,
                        parameters: Parameters) -> Numeric:
        return state

    def explicit_solution(self, t, parameters):
        a, x0 = parameters
        return np.exp(-a * t) * x0

    def dxdp(self, t, parameters):
        a, x0 = parameters
        return [
            -t * a * np.exp(-a*t) * x0,
            np.exp(-a * t)
        ]

    def forwards(self, t,  p, dp):
        x = self.explicit_solution(t, p)
        a, x0 = p
        return [
            -t * a * x * dp[0] + x * dp[1] / x0
        ]


class TestSymbolicIntegrator:

    def test_init(self):
        block = LinearScalarEquation()

        with SolverContext(model=block, t_final=10) as solver:

            f = solver.get_integrator()
            t = 2
            params = [1, 1]
            result = float(f(t, params)[0][0])
            expected_result = block.explicit_solution(t, params)

            assert abs(result - expected_result) < 1e-4

    def test_finite_differences(self):
        block = LinearScalarEquation()

        with SolverContext(model=block, t_final=10) as solver:
            f = solver.get_integrator()
            t = 2
            params = [1, 1]
            param_p = [1.1, 1]
            dparams = [0.1, 0]
            result = (f(t, param_p) - f(t, params))

            expected = block.forwards(t, params, dparams)
            assert abs(result[0][0] - expected) < 1e-2

    def test_autodiff(self):
        block = LinearScalarEquation()

        with SolverContext(model=block, t_final=10) as solver:
            f = solver.get_integrator()
            t = 2
            params = [1, 1]
            dparams = [0.1, 0]

            result = f.forwards(t, params, dparams)

            expected = block.forwards(t, params, dparams)
            assert abs(result[0][0] - expected) < 1e-2

