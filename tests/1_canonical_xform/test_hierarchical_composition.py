"""Test cases for hierarchical composition."""

from sysopt import Metadata
from sysopt.block import Composite
from sysopt.blocks.builders import FullStateOutput, InputOutput
from sysopt.blocks.block_operations import create_functions_from_block
from sysopt.blocks.common import Gain

def build_mock_system():
    inner_metadata = Metadata(
        states=['x'],
        inputs=['u'],
        parameters=['x0']
    )

    def x0(p):
        return p

    def f(t, x, u, p):
        return -x[0] + u[0]

    block_inner_1 = FullStateOutput(inner_metadata, f, x0)

    def identity(t, u, p):
        return u

    block_inner_2 = InputOutput(Metadata(inputs=['u'], outputs=['y']),
                                identity)

    composite_inner = Composite()
    composite_inner.components = [block_inner_1, block_inner_2]
    composite_inner.wires = [
        (block_inner_2.outputs, block_inner_1.inputs),
        (block_inner_1.outputs, composite_inner.outputs),
        (composite_inner.inputs, block_inner_2.inputs)
    ]
    return composite_inner


def test_inner_system():
    model = build_mock_system()

    x0, f, g, h, tables = create_functions_from_block(model)

    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


def test_build_outer_system():

    outer_gain = Gain(channels=1)
    model = build_mock_system()

    outer_composite = Composite()
    outer_composite.components = [outer_gain, model]
    outer_composite.wires = [
        (outer_gain.outputs, model.inputs),
        (model.outputs, outer_composite.outputs),
        (model.outputs, outer_gain.inputs)
    ]

    x0, f, g, h, tables = create_functions_from_block(outer_composite)

    assert len(outer_composite.parameters) == 2
    p = [3, 5]
    x0_value, = x0(p)
    assert x0_value == p[0]
    domain = [1, 1, 3, 0, 2]

    f_values, = f(0, 2, [7, 11, 13], None, p)

    assert f_values == -2 + 7
