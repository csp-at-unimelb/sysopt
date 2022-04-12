from sysopt.symbolic import (
    Variable, is_symbolic, list_symbols, Parameter,
    is_temporal, ExpressionGraph, lambdify
)

from sysopt.symbolic.symbols import get_time_variable
import sysopt.symbolic.scalar_ops as scalar_ops

from sysopt.blocks import Gain, Oscillator
from sysopt.block import Composite

import numpy as np


def test_variables_api():

    d = Variable()
    d2 = Variable()
    assert is_symbolic(d)
    expressions = [d + 2, d - 2, -d, 2-d, 2+d, d/2, 2 / d, 2 * d, d*2, d + d2]

    for expression in expressions:
        assert is_symbolic(expression)
        assert d in list_symbols(expression)

    for op in scalar_ops.unary:
        result = op(d)
        assert is_symbolic(result)
        assert d in list_symbols(result)


def test_parameter_api():
    second_block = Gain(2)
    block = Gain(1)
    param = Parameter(block, 0)

    host = Composite()
    host.components = [block, second_block]

    assert param is not None
    source, slce = param.get_source_and_slice()

    assert source is block
    assert (slce.start, slce.stop, slce.step) == (0, 1, None)
    assert param.name in host.parameters
    assert host.parameters.index(param.name) == 0
    assert param.shape == (1, )


class TestSignalApi:
    def test_create_and_evaluate(self):
        source = Oscillator()

        t = get_time_variable()
        sig = source.outputs(t)
        assert sig is not None
        assert sig.port is source.outputs
        expr = sig(0)

        assert expr is not None
        assert expr is not sig

        assert isinstance(expr, ExpressionGraph)

    def test_identity(self):
        gain = Gain(2)
        t = get_time_variable()
        sig_0 = gain.outputs[0](t)
        sig_1 = gain.outputs[1](t)

        assert sig_0.symbols() == sig_1.symbols()


def test_is_temporal():
    var = Variable()
    source = Oscillator()
    t = get_time_variable()
    sig = source.outputs(t)
    param = Parameter(source, 0)

    assert not is_temporal(var)
    assert not is_temporal(param)
    assert is_temporal(sig)

    assert not is_temporal(var + param)
    assert is_temporal(var + sig)
    assert is_temporal(param + sig)

    assert not is_temporal(sig(0))
    assert not is_temporal(sig(1))


def test_evaluate_graph():

    def y(tau):
        return np.exp(tau)

    var = Variable()
    source = Oscillator()
    t = get_time_variable()
    sig = source.outputs(t)
    param = Parameter(source, 0)

    expression = 1 + var * sig(1) + param
    values = {
        var: 2,
        sig: y,
        param: 3
    }
    expected_result = 1 + 2 * np.exp(1) + 3

    assert expression.symbols() == {var, sig, param}

    result = expression.call(values)
    assert result == expected_result


def test_inequality_to_function():
    var = Variable()
    var_inequality = var <= 1

    assert var_inequality.call({var: 2}) < 0
    assert var_inequality.call({var: 1}) == 0
    assert var_inequality.call({var: 0}) > 0

    f = lambdify(var_inequality.to_graph(), [var])
    assert f(2) < 0
    assert f(1) == 0
    assert f(0) > 0


def test_supremum_to_function():
    t = get_time_variable()
    source = Oscillator()
    sig = source.outputs(t)
    param = Parameter(source, 0)

    supremum = sig(t) < param

    def y(t_):
        return np.exp(-t_)

    expr = supremum.call({param: 2, sig: y})

    assert expr.symbols() == {t}
    result = expr.call({t: 0})
    assert result > 0

# Optimisation problem setup
# 1. Build model
# 2. Add quadrature for loss function
# 3. Add constraint dynamics


