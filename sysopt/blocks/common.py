"""Commonly used blocks for model building."""

from sysopt import Block, Signature, Metadata
from numpy import cos


class Gain(Block):
    r"""Block for a simple gain stage.

    For each channel (indexed by i), the input-output relationship
    is given by

    math::

        y[i] = p[i] * u[i], i = 0,..., channels - 1

    where :math:`u` are the inputs, :math:`y` are the outputs and :math:`p`
    are the parameters.

    Args:
        channels: Number of gain channels to provide.

    """
    def __init__(self, channels):
        sig = Signature(inputs=channels,
                        outputs=channels,
                        parameters=channels)

        super().__init__(sig)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return [gain * signal for signal, gain in zip(inputs, parameters)]


class Mixer(Block):
    """Provides the output equal to the sum of the inputs.

    Args:
        inputs: number of input channels.

    """
    def __init__(self, inputs):
        sig = Signature(
            inputs=inputs,
            outputs=1,
            state=0,
            parameters=0
        )
        super().__init__(sig)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return sum(inputs),


class ConstantSignal(Block):
    r"""Output constant on each channel.

    For each channel, a constant signal :math:`y[i]` is output equal to the
    corresponding parameter value :math:`p[i]`. That is,

    math::
        y[i](t) = p[i], \text{for all} t

    Args:
        outputs: The number of output channels.

    """
    def __init__(self, outputs):
        sig = Signature(outputs=outputs, parameters=outputs)
        super().__init__(sig)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return parameters


class Oscillator(Block):
    """Cosine oscillator with the given frequency and phase."""

    def __init__(self):
        metadata = Metadata(
            parameters=['frequency', 'phase'],
            outputs=['signal']
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        freq, phase = parameters
        return cos(t * freq + phase),


class LowPassFilter(Block):
    """First order low-pass filter."""
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
