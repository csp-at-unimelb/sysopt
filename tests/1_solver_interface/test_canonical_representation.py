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
        return [1, 1]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        v, x = states
        omega, gamma = parameters

        return [- 2 * omega * gamma * v - x * omega ** 2, v]

    def compute_residuals(self, t, states, algebraics, inputs, parameters):
        v, x = states
        h, = algebraics
        return h - x ** 2

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states[1]


def test_flatten_system():
    block = DampedHarmonicOscillator()


