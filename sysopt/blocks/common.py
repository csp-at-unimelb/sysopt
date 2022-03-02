from sysopt import Block, Signature, Metadata
from numpy import cos


class Gain(Block):
    def __init__(self, channels):
        sig = Signature(inputs=channels,
                        outputs=channels,
                        state=0,
                        parameters=channels)

        super().__init__(signature=sig)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        return [gain * signal for signal, gain in zip(inputs, parameters)]


class Mixer(Block):
    def __init__(self, inputs):
        sig = Signature(
            inputs=inputs,
            outputs=1,
            state=0,
            parameters=0
        )
        super().__init__(sig)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        return sum(inputs),


class ConstantSignal(Block):
    def __init__(self, outputs):
        sig = Signature(outputs=outputs, parameters=outputs)
        super().__init__(sig)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        return parameters


class Oscillator(Block):
    def __init__(self):
        metadata = Metadata(
            parameters=['frequency', 'phase'],
            outputs=['signal']
        )
        super(Oscillator, self).__init__(metadata)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        freq, phase = parameters
        return cos(t * freq + phase),


class LowPassFilter(Block):
    def __init__(self):
        metadata = Metadata(
            parameters=['cutoff frequency'],
            inputs=['input'],
            outputs=['output'],
            state=['state']
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        return 0,

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        x, = state
        w, = parameters
        u, = inputs
        return (u - x) / w,

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        x, = state
        return x,
