import numpy as np
import random

import sympy

from sysopt import Metadata
from sysopt.modelling.block import Composite, Block
from sysopt.blocks import Gain, Oscillator, LowPassFilter
from sysopt.problems import SolverContext, Parameter, create_parameter_map
from sysopt.symbolic import Variable

eps = 1e-4


def ode_solution(t, w_osc=1, phase=0, w_coff=1, gain=1, x0=0):
    alpha = 1 / w_coff
    mag = alpha ** 2 + w_osc ** 2
    out = alpha ** 2 * np.cos(w_osc * t + phase) / mag
    out += alpha * w_osc * np.sin(w_osc * t + phase) / mag
    out += np.exp(-alpha * t) * (
        x0
        - alpha ** 2 * np.cos(phase) / mag
        - w_osc * alpha * np.sin(phase) / mag
    )
    return gain * out


def expected_output(t, w):
    return ode_solution(t, w_coff=w)


def expected_quadrature(t, w_coff, w_osc=1, phase=0, x0=1):

    t_s = sympy.S('t')
    alpha = 1 / w_coff
    mag = alpha ** 2 + w_osc ** 2
    out = alpha ** 2 * sympy.cos(w_osc * t_s + phase) / mag
    out += alpha * w_osc * sympy.sin(w_osc * t_s + phase) / mag
    out += sympy.exp(-alpha * t_s) * (
        x0
        - alpha ** 2 * sympy.cos(phase) / mag
        - w_osc * alpha * sympy.sin(phase) / mag
    )
    integrand = out ** 2
    c = integrand.evalf(10, {t_s: 0})
    soln = sympy.integrate(integrand, (t_s, 0, t)) - c

    return float(soln)


class FilteredOscMockUnconstrained(Block):
    def __init__(self):
        metadata = Metadata(
            states=['filter'],
            outputs=['post gain'],
            parameters=['osc frequency',
                        'phase',
                        'cutoff frequency',
                        'gain']
        )
        super().__init__(metadata, 'FilteredOsc')

    def initial_state(self, parameters):
        return 0

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        x, = states
        w_osc, phi, w_cf, _ = parameters
        return (np.cos(w_osc * t + phi) - x) / w_cf

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        *_, gain = parameters
        x, = states
        return gain * x


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

    return model, constants, expected_output, expected_quadrature


class TestParameterMap:
    """parameter map
        inputs:
        - a model,
        - an existing set of constants
        - an optional terminal time

       outputs:
        - a list of parameter p'
        - a function that maps p' -> t_final, p_actual
    """

    def test_generates_identity_map_with_fixed_time_no_args(self):
        model, *_ = build_example()

        expected_params = [Parameter(model, p) for p in model.parameters]

        t_final = 10
        _, params, t_map, p_map = create_parameter_map(model, {}, t_final)
        assert len(params) == len(expected_params)
        results = list(zip(expected_params, params))

        assert all(expected_p == p for expected_p, p in results),\
            str(results)

        p_test = [random.random() for _ in params]

        t_result, p_result = t_map(p_test), p_map(p_test)
        assert t_result == t_final
        assert p_test == p_result

    def test_generates_constant_map_with_variable_time(self):
        model, *_ = build_example()
        constants = {p: random.random() for p in model.parameters}

        t_final = Variable('t_final')
        _, params, t_map, p_map = create_parameter_map(model, constants, t_final)

        assert params == [t_final]
        t_test = random.random()
        t_result = t_map(t_test)
        assert t_result == t_test
        p_result = p_map(t_test)
        assert p_result == list(constants.values())

    def test_mixed_map(self):
        model, constants, *_ = build_example()
        t_final = Variable('t_final')
        _, params, t_map, p_map = create_parameter_map(model, constants, t_final)

        assert params[0] == t_final
        values = [0, 1]
        result = p_map(values)

        assert result == [1, 0, 1, 1]


class TestSolverUnconstrained:

    def test_builds_solver(self):
        block = FilteredOscMockUnconstrained()
        constants = {
            block.parameters[0]: 1,
            block.parameters[1]: 0,
            block.parameters[3]: 1
        }

        t_f = 2
        test_params = [1, 0, 1, 1]

        with SolverContext(block, t_f, constants) as solver:
            integrator = solver.get_integrator(resolution=150)
            x0 = integrator.x0(test_params)
            assert x0.shape == (2, 1)
            x0 = [x0[0, 0], x0[1, 0]]
            assert x0 == [0, 0]

    def test_solution(self):
        block = FilteredOscMockUnconstrained()
        constants = {
            block.parameters[0]: 1,
            block.parameters[1]: 0,
            block.parameters[3]: 1
        }

        t_f = 2
        res = 50
        with SolverContext(block, t_f, constants) as solver:
            w = 1
            y = solver.integrate(w, t_f, resolution=res)
            error = [
                abs(expected_output(y.t[i], w) - y.x[i])
                for i in range(res)
            ]
            assert all(list(e < 1e-4 for e in error)), \
                "Error performing integration - solution points dont match" \
                "analytic result"

    def test_quadrature(self):
        model, constants, output, quad = build_example()

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

            # this should bind the y to the problems context via t
            y = model.outputs(t)    # a symbol for y at time t

            squared = solver.integral(y ** 2)
            # this should add a quadrature to the state variables.
            # in particular, we should have
            # dot{q_0} = y^2, q(0) = 0
            # stored somewhere in the problems workspace

            # we should be able to check that this is now a function
            # with 2 arguments: time t and

            w = 1
            y, q = solver.integrate(w, t_f, resolution=150)
            result = squared(t_f, 1)

            # calling

            # solution should be given by
            expected_soln = quad(t_f, w)
            assert abs(result - expected_soln) < 1e-4


class TestSolverCompositeModel:
    def test_solution(self):
        model, constants, output, quad = build_example()

        t_f = 10
        with SolverContext(model, t_f, constants) as solver:
            w = 1
            y = solver.integrate(w, t_f)

            error = [
                abs(output(y.t[i], w) - y(y.t[i]))
                for i in range(y.t.shape[0])
            ]
            assert all(list(e < 1e-4 for e in error)), error

    def test_quadrature(self):
        model, constants, output, quad = build_example()

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

            # this should bind the y to the problems context via t
            y = model.outputs(t)    # a symbol for y at time t

            squared = solver.integral(y ** 2)
            # this should add a quadrature to the state variables.
            # in particular, we should have
            # dot{q_0} = y^2, q(0) = 0
            # stored somewhere in the problems workspace

            # we should be able to check that this is now a function
            # with 2 arguments: time t and

            w = 1
            soln = squared(t_f, 1)
            # calling

            # solution should be given by
            expected_soln = quad(t_f, w)
            assert abs(soln - expected_soln) < 1e-4
