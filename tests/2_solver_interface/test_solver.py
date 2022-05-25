import numpy as np
import pytest
import random
from sysopt.block import Composite
from sysopt.blocks import Gain, Oscillator, LowPassFilter
from sysopt.solver import SolverContext, Parameter, create_parameter_map
from sysopt.symbolic import Variable, find_param_index_by_name
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

    def quadrature(t, w):
        # Analytically determined via Wolfram Alpha, or Sympy
        scaling = 4 * (1 + w ** 2) ** 2

        def integral(tau):
            return (
              2 * (tau * (1 + w**2) + w)
              + (w**2 - 1) * math.sin(2 * tau)
              - 2 * w * math.cos(2 * tau)
            ) / scaling

        return integral(t) - integral(0)

    return model, constants, output, quadrature


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
        params, t_map, p_map = create_parameter_map(model, {}, t_final)
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
        params, t_map, p_map = create_parameter_map(model, constants, t_final)

        assert params == [t_final]
        t_test = random.random()
        t_result = t_map(t_test)
        assert t_result == t_test
        p_result = p_map(t_test)
        assert p_result == list(constants.values())

    def test_mixed_map(self):
        model, constants, *_ = build_example()
        t_final = Variable('t_final')
        params, t_map, p_map = create_parameter_map(model, constants, t_final)

        assert params[0] == t_final
        values = [0, 1]
        result = p_map(values)

        assert result == [1, 0, 1, 1]


class TestSolver:

    def test_init(self):
        model, constants, output, quad = build_example()

        t_f = 10
        with SolverContext(model, t_f, constants) as solver:
            fs = solver.flattened_system

        assert False


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

            # this should bind the y to the solver context via t
            y = model.outputs(t)    # a symbol for y at time t

            squared = solver.integral(y ** 2)
            # this should add a quadrature to the state variables.
            # in particular, we should have
            # dot{q_0} = y^2, q(0) = 0
            # stored somewhere in the solver workspace

            # we should be able to check that this is now a function
            # with 2 arguments: time t and

            w = 1
            y_f, q_f = solver.integrate(parameters=w, t_final=t_f)
            error = [
                output(t_i, 1) - y_f(t_i) for t_i in np.linspace(0, t_f, 50)
            ]

            soln = squared(t_f, 1)
            # calling

            # solution should be given by
            expected_soln = quad(t_f, w)
            assert abs(soln - expected_soln) < eps

