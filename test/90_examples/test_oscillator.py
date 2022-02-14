#
#
from codesign import DenseArray, Parameter, System, Vector
from codesign.blocks.common import LinearSystem, Gain
from codesign.functions import sum_of_squares, time_integral
from codesign.problem import Minimise, Solver
import numpy as np


def test_oscillator():
    k = Parameter(value=1, bounds=(1, 100))
    w = Parameter(value=2, bounds=(1, 4))

    plant = LinearSystem(A=DenseArray([[0, -w**2], [1, 0]]),
                         B=DenseArray([[1], [0]]),
                         C=np.eye(2),
                         x0=Vector([0, 1]))

    controller = Gain(k)

    system = System(
        components=[plant, controller],
        wires=[(plant.inputs, controller.outputs), (plant.outputs[1], controller.inputs)]
    )

    signature = system.signature
    assert signature.inputs == 2
    assert signature.state == 2
    assert signature.outputs == 3
    assert signature.parameters == 2

    x = plant.outputs
    u = controller.outputs
    loss = time_integral(sum_of_squares(x))

    problem = Minimise(loss, system, constraints=[])

    solver = Solver()
    sol_n = solver.solve(problem, window=[0, 10])
    t_f = sol_n.window[-1]
    assert sol_n.value
    assert sol_n.argmin[w]
    assert sol_n.argmin[k]
    assert sol_n.argmin[x](t_f).shape == (2,)
    assert sol_n.argmin[u](t_f).shape == (1,)
