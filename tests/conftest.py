import pytest
import sys
from os.path import dirname as d
from os.path import abspath
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)


@pytest.fixture
def linear_model():
    from sysopt import Composite, Metadata, FullStateOutput
    from sysopt.blocks import ConstantSignal
    import numpy as np

    model = Composite(name='Test Model')
    # build a LQR model
    #
    plant_metadata = Metadata(
        inputs=['u'],
        states=['x_0', 'x_1']
    )
    A = np.array([[0, 1],
                  [-1, 0]], dtype=float)
    B = np.array([[1], [0]], dtype=float)

    def f(t, x, u, _):
        return A @ x + B @ u

    def x0(_):
        return np.array([0, 1])

    plant = FullStateOutput(
        dxdt=f,
        metadata=plant_metadata,
        x0=x0,
        name='plant'
    )
    control = ConstantSignal(['u'])
    model.components = [plant, control]
    model.declare_outputs(['x_0', 'x_1', 'u'])
    model.wires = [
        (control.outputs, plant.inputs),
        (control.outputs[0], model.outputs[2]),
        (plant.outputs[0], model.outputs[0]),
        (plant.outputs[1], model.outputs[1])
    ]
    return model