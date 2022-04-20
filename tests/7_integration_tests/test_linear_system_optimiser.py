"""
Problem Setup


Minimise
    ||x||_Q + <p|x> + ||u||

With respect to K

Such that:
    |u| < 1
    dx = Ax + Bu
    u = Kx

"""
from sysopt import Signature, Composite
from sysopt.solver import SolverContext, Parameter
from sysopt.symbolic import time_integral
from sysopt.blocks.builders import FullStateOutput, InputOutput
import numpy as np


def build_lqr_model():

    A = np.array(
        [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        dtype=float
    )

    B = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float).T

    def x0(_):
        return np.array([[0, 0, 1, 1]], dtype=float).T

    def dxdt(t, x, u, p):
        return A @ x + B @ u

    def lqr(t, x, p):
        k = np.array(p).reshape((4, 2))
        return k @ x

    plant = FullStateOutput(Signature(inputs=2, states=4), dxdt=dxdt, x0=x0)
    controller = InputOutput(Signature(inputs=4, outputs=2, parameters=8), lqr)

    model = Composite()
    model.components = [plant, controller]
    model.wires = [
        (plant.outputs, controller.inputs),
        (controller.outputs, plant.inputs),
        (plant.outputs, model.outputs[0:4]),
        (controller.outputs, model.outputs[4:6])
    ]

    return model


def test_model_assembly():
    model = build_lqr_model()
    assert len(model.outputs) == 6
    _, controller = model.components
    parameters = Parameter.from_block(controller)
    initial_values = [0.01] * len(parameters)
    t_f = 10
    Q_x = np.eye(4)
    Q_u = np.eye(2)

    with SolverContext(model, t_f) as solver:
        t = solver.t
        u = model.outputs[4:6](t)
        x = model.outputs[0:4](t)

        loss = time_integral(x.T @ Q_x @ x + u.T @ Q_u @ u)
        constraints = [
            x[0: 2].T @ x[0: 2] < 2,
            u.T @ u < 1
        ]

        problem = solver.problem(parameters, loss, constraints)

        # loss function should become a quadrature and terminal cost
        # constraints should become quadratures and barrier functions

        # Address:
        # - How to index the added variables
        # - How to index / identify the added functions
        # - How to evaluate the added variables / functions
        # - How to make sure constraints are mapped into barrier functions
        # - How to certify that an initial solutions is not feasible.
        # - How to get the adjoint sensitivity of the loss function
        #   with respect to the parameters.
