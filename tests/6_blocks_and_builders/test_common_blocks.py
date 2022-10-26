import numpy as np
from sysopt.modelling.block import Composite
from sysopt.problems import SolverContext
from sysopt.blocks.common import HighPassFilter, Oscillator, DifferentialAmplifier


def test_highpass_filter_as_differentiator():

    osc_1 = Oscillator(name='Osc_1')
    osc_2 = Oscillator(name='Osc_2')
    flter = HighPassFilter(name='HPF')
    diffamp = DifferentialAmplifier()
    model = Composite(name='Differentiator')

    model.components = [osc_1, osc_2, flter, diffamp]
    model.wires = [
        (osc_1.outputs, flter.inputs),
        (flter.outputs, diffamp.inputs[0]),
        (osc_2.outputs, diffamp.inputs[1]),
        (diffamp.outputs, model.outputs[0])
    ]
    osc_freq = 1        # hz
    hpf_freq = 0.01     # hz

    constants = {
        osc_1.parameters[0]: osc_freq,
        osc_2.parameters[0]: osc_freq,
        osc_1.parameters[1]: 0,
        osc_2.parameters[1]: np.pi/2,   # trig diff -> phase shift of pi/2
        flter.parameters[0]: hpf_freq,
        diffamp.parameters[0]: 0,
        diffamp.parameters[1]: 0
    }

    with SolverContext(model, 10, constants) as solver:
        soln = solver.integrate()
    x = soln.x.full().ravel()[1:]
    error = x.T @ x / len(x)    # ~ \int_0^t |x(x)|^2 dt
    assert error < 1e-3
