import pytest

from sysopt.block import Composite
from sysopt.blocks import Gain, Oscillator, LowPassFilter
from sysopt.solver import SolverContext, Parameter
from sysopt.symbolic import time_integral
import math

eps = 1e-4


def build_example():

    osc = Oscillator()
    gain = Gain(channels=1)
    lpf = LowPassFilter()
    model = Composite()

    model.components = [osc, lpf, gain]
    model.wires = [
        (osc.outputs, lpf.inputs),
        (lpf.outputs, gain.inputs),
        (gain.outputs, model.outputs)
    ]
    constants = {
        osc.parameters[0]: 1,
        osc.parameters[1]: 0,
        gain.parameters[0]: 1
    }

    def output(t, w):
        return (w * math.cos(t) + math.sin(t))/(w**2 + 1)

    return model, constants, output

@pytest.mark.skip
def test_integrate():
    model, constants, output = build_example()

    t_f = 10
    with SolverContext(model, t_f, constants) as solver:
        parameterised_solution = solver.get_symbolic_integrator()
        x_t1 = parameterised_solution(0.5)
        x_t2 = parameterised_solution(1)

    for t in range(10):
        result_1 = x_t1(t)
        expected_1 = output(t, 0.5)

        result_2 = x_t2(t)
        expected_2 = output(t, 1)

        assert abs(result_1 - expected_1) < 1e-4
        assert abs(result_2 - expected_2) < 1e-4

@pytest.mark.skip
def test_quadrature():
    osc = Oscillator()
    gain = Gain(channels=1)
    lpf = LowPassFilter()
    model = Composite()

    model.components = [osc, lpf, gain]
    model.wires = [
        (osc.outputs, lpf.inputs),
        (lpf.outputs, gain.inputs),
        (gain.outputs, model.outputs)
    ]

    constants = {
        osc.parameters[0]: 1,
        osc.parameters[1]: 0,
        lpf.parameters[0]: 1
    }
    t_f = 10
    with SolverContext(model, t_f, constants) as solver:
        # the Solver object should contain a function
        # that represents the solution to the ode
        # with arguments
        #  - t in [0, t_f]
        #  - p in R^1

        # we should have a set identifying the un-assigned variables
        # 3 notions of 't'
        # - 't' as an argument
        # - 't' as a free variable symbol
        # - 't' as the independent variable of an integration scheme
        t = solver.t            # a symbol for t in [0,t_f]

        # this should bind the y to the solver context via t
        y = model.outputs(t)    # a symbol for y at time t

        squared = time_integral(y ** 2, solver)
        # this should add a quadrature to the state variables.
        # in particular, we should have
        # dot{q_0} = y^2, q(0) = 0
        # stored somewhere in the solver workspace

        # we should be able to check that this is now a function
        # with 2 arguments: time t and

        soln = squared(10, 1)
        # calling

        # solution should be given by
        # integral_0^10 cos^2(t) dt = 5 + sin(20)/4
        expected_soln = 5 + math.sin(20) / 4
        assert abs(soln - expected_soln) < eps

