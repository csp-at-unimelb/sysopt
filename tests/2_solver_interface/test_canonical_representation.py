from sysopt.solver import SolverContext
from sysopt.symbolic import Parameter
from sysopt import Metadata
from sysopt.block import Block
from sysopt.blocks.block_operations import create_functions_from_block
# Canonical representation is made up of
# - Domain (number of states)
# - [F(t, \zeta), H(t,\zeta)] = M\dot{\zeta}}
# - G(t, \zeta) = y


# For an optimisation problem we also have:
# - rho (regularisation parameters)
# - \dot{Q} = Phi(y, rho)
#

metadata = Metadata(
    states=['velocity', 'position'],
    constraints=['height'],
    parameters=['frequency', 'damping'],
    outputs=['position']
)


class DampedHarmonicOscillator(Block):
    def __init__(self):
        super().__init__(metadata)

    def initial_state(self, parameters):
        return [1, 0]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        v, x = states
        omega, gamma = parameters

        return [- 2 * omega * gamma * v - x * omega ** 2, v]

    def compute_residuals(self, t, states, algebraics, inputs, parameters):
        v, x = states
        h, = algebraics
        return h - x ** 2,

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states[1],


# def test_add_quadratures():
#     """
#
#     """
#     from sysopt.symbolic import time_integral
#     block = DampedHarmonicOscillator()
#
#     t_final = 10
#     constants = {block.parameters[0]: 0.1}
#     frequency = Parameter(block, 1)
#     with SolverContext(block, t_final, constants) as solver:
#         y = block.outputs(solver.t)
#         q_dot = y ** 2
#         quadrature = time_integral(q_dot)
#
#         # integrate solution


