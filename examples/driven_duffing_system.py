import numpy as np

from sysopt import Block, Metadata, Composite
from sysopt.blocks import Gain, Oscillator


class DuffingComponent(Block):
    def __init__(self,
                 initial_position=0,
                 initial_velocity=0):
        metadata = Metadata(
            inputs=['force'],
            state=['position', 'velocity'],
            parameters=['damping', 'stiffness', 'nonlinearity'],
            outputs=['position', 'velocity']
        )
        super().__init__(metadata)

        self.x0 = [initial_position, initial_velocity]

    def initial_state(self, parameters):
        return self.x0

    def compute_dynamics(self, t, state, algebraic, inputs, parameters):
        delta, alpha, beta = parameters
        x, dx = state
        u, = inputs
        return [
            dx,
            -delta * dx - alpha * x - beta * x ** 3 + u
        ]

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        x, dx = state
        return [x, dx]


class DuffingSystem(Composite):
    def __init__(self):
        self.oscillator = Oscillator()
        self.resonator = DuffingComponent()
        self.gain = Gain(channels=1)

        wires = [
            (self.oscillator.outputs, self.gain.inputs),
            (self.gain.outputs, self.resonator.inputs),
            (self.gain.outputs[0], self.outputs[1]),
            (self.oscillator.outputs[0], self.outputs[0])
        ]

        super().__init__(
            components=[self.resonator, self.gain, self.oscillator],
            wires=wires
        )


def simulate():
    duffing_system = DuffingSystem()
    default_parameters = [0.2, -1, 1, 0.3, 1, 0]

    # get an integrator
    model = get_flattened_system(duffing_system)
    func = backend.integrator(model)

    assert len(func.inputs) == 0
    assert len(func.outputs) == 2
    assert len(func.parameters) == len(default_parameters)

    T = np.linspace(0, 10, 25)
    X = np.zeros(shape=(len(T), 2))
    for i, t in enumerate(T):
        X[i, :] = func.compute_outputs(10, [], [], default_parameters)


if __name__ == '__main__':
    simulate()