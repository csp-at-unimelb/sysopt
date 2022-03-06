import numpy as np

from sysopt.blocks.common import Gain, LowPassFilter, Oscillator
from sysopt.block import Composite
from sysopt.backends.casadi import CasadiBackend, CasadiVector
from sysopt.backend import get_default_backend


def test_vector_class():
    v = CasadiVector('v', 2)

    v0, v1 = v
    assert v0 == v[0]
    assert v1 == v[1]


def test_create_variables():
    ctx = CasadiBackend()
    block = Gain(channels=2)

    x, z, u, p = ctx.get_or_create_variables(block)
    assert not x
    assert not z
    assert u.shape == (2, 1)
    assert p.shape == (2, 1)

    lpf = LowPassFilter()

    x1, z1, u1, p1 = ctx.get_or_create_variables(lpf)

    assert x1.shape == (1, 1)
    assert u1.shape == (1, 1)
    assert p1.shape == (1, 1)


def test_create_flattened_system_leaf():
    ctx = get_default_backend()

    block = Gain(2)
    block2 = Gain(1)
    flat_block_1 = ctx.get_flattened_system(block)
    assert len(ctx._variables) == 1
    _ = ctx.get_flattened_system(block)
    assert len(ctx._variables) == 1
    flat_block_2 = ctx.get_flattened_system(block2)
    assert len(ctx._variables) == 2
    assert block2 in ctx._variables

    assert not flat_block_1.X
    assert not flat_block_1.Z
    assert flat_block_1.U.shape == (2, 1)
    assert flat_block_1.P.shape == (2, 1)
    assert not flat_block_1.f
    assert flat_block_1.g.shape == (2, 1)
    assert not flat_block_1.h

    flattened_system = flat_block_1 + flat_block_2

    assert not flattened_system.X
    assert not flattened_system.Z
    assert flattened_system.U.shape == (3, 1)
    assert flattened_system.P.shape == (3, 1)
    assert flattened_system.g.shape == (3, 1)


def test_create_flattened_system_composite():
    ctx = get_default_backend()

    osc = Oscillator()
    lpf = LowPassFilter()
    # corresponds to
    # cos(t) = x' + x
    # y = x
    # with x0 = 0
    # solution should be
    # y(t) = 0.5(sin(t) + cos(t) - exp(-t))

    def y(t):
        return 0.5 * (np.sin(t) + np.cos(t) - np.exp(-t))

    assert len(lpf.inputs) == 1
    assert len(lpf.outputs) == 1

    composite = Composite(
        components=[osc, lpf],
        wires=[(osc.outputs, lpf.inputs)]
    )
    assert len(lpf.inputs) == 1

    composite.wires += [(lpf.outputs, composite.outputs)]

    flattened_system = ctx.get_flattened_system(composite)

    assert flattened_system.X.shape == (1, 1)
    assert flattened_system.Z.shape == (1, 1)
    assert flattened_system.P.shape == (3, 1)
    assert not flattened_system.U
    assert flattened_system.f.shape == (1, 1)
    assert flattened_system.g.shape == (1, 1)
    assert flattened_system.h.shape == (1, 1)
    assert flattened_system.X0.shape == (1, 1)
    f = ctx.integrator(flattened_system)

    p = [1, 0, 1]
    y_sol = f(10, p)

    for i in range(10):
        assert abs(float(y_sol(i)) - y(i)) < 1e-4
