import pytest

from sysopt.blocks.discrete_controllers import PIDController
import sysopt.symbolic as symbolic
import numpy as np
from sysopt.blocks.common import Oscillator, LowPassFilter, DifferentialAmplifier
from sysopt import Composite
from sysopt.problems import SolverContext
from sysopt.modelling.block import DiscreteBlock


def test_pid_isolated():

    pid = PIDController(
        error_signals=['a', 'b'],
        name='pid test',
        clock_hz=10
    )

    assert pid is not None
    assert pid.frequency == 10
    assert pid.dt == 0.1
    assert pid.signature.outputs == 2
    assert pid.signature.states == 6
    assert pid.signature.inputs == 2
    k_d = pid.parameters[0::3]
    assert k_d == ['pid test/a:K_d', 'pid test/b:K_d'], k_d
    k_p = pid.parameters[1::3]
    assert k_p == ['pid test/a:K_p', 'pid test/b:K_p']
    k_i = pid.parameters[2::3]
    assert k_i == ['pid test/a:K_i', 'pid test/b:K_i']

    gains = np.arange(len(pid.parameters))
    x0 = pid.initial_state(gains)
    assert x0 == [0] * pid.signature.states

    x = np.array([1] * pid.signature.states, dtype=float)

    inputs = np.array([-1, -2], dtype=float)
    x_next = pid.compute_state_transition(
        t=0,
        states=np.array(x, dtype=float),
        algebraics=None,
        inputs=inputs,
        parameters=gains
    )

    assert (x_next[-2:] == x[2:-2]).all(), 'Last entries should be shuffled down'
    assert (x_next[2:-2] == inputs).all(), 'Middle entries should be the inputs'

    assert isinstance(pid, DiscreteBlock)


def test_pid_symbolic_call():

    pid = PIDController(
        error_signals=['a', 'b'],
        name='pid test',
        clock_hz=10
    )
    t = symbolic.get_time_variable()
    x = symbolic.symbolic_vector('x', pid.signature.states)
    z = symbolic.symbolic_vector('z', pid.signature.constraints)
    u = symbolic.symbolic_vector('u', pid.signature.inputs)
    p = symbolic.symbolic_vector('p', pid.signature.parameters)

    x_next = pid.compute_state_transition(t, x, z, u, p)

    assert x_next.is_symbolic
    symbols = x_next.symbols()
    assert x in symbols
    assert u in symbols
    assert p in symbols


class TestSystemWithPID:
    def test_flattening(self):

        # reference signal: v = cos(wt)          [OSC]
        # plant: dot{x} = (u - x) / gamma        [LPF]
        # Error:  e = v - x                      [DIFF]
        # controller: u = pid(e)
        # output: y = x

        #

        osc = Oscillator()
        plant = LowPassFilter()
        error = DifferentialAmplifier()
        controller = PIDController()
        test_fixture = Composite()

        test_fixture.components = [
            osc, plant, error, controller
        ]
        test_fixture.wires = [
            (osc.outputs, error.inputs[0]),
            (error.outputs, plant.inputs),
            (plant.outputs, controller.inputs),
            (controller.outputs, error.inputs[1]),
            (controller.outputs, test_fixture.outputs[1]),
            (plant.outputs, test_fixture.outputs[0])
        ]

        constants = {
            osc.parameters[0]: 1,    # freq
            osc.parameters[1]: 0,    # phase
            plant.parameters[0]: 1,  # cutoff freq
            error.parameters[0]: 0,  # 0 dB gain
            error.parameters[1]: 0   # 0 bias
        }
        t_final = 10
        with SolverContext(test_fixture, t_final, constants) as solver:
            flat_system = solver.flattened_system
            assert flat_system.state_transitions is not None
            assert len(flat_system.state_transitions) == 1
            freq, f, h = flat_system.state_transitions[0]
            assert freq == 100
            assert h is None

            with pytest.raises(NotImplementedError):
                soln = solver.get_integrator()









