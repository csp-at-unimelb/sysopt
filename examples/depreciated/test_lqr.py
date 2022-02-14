from codesign.blocks.helpers import ode_builder, function_wrapper\
#, quad_form, sum_of_squares, function, integral
from codesign.core import Problem, System
import numpy as np
from codesign.perf_tools import Timer
from scipy.linalg import solve_continuous_are


def test_lqr():
    A = np.array([[0, -1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, -1]], dtype=float)
    B = np.array([0.1, 0.2, 0, 0.1], dtype=float)

    def dxdt(t, x, u):
        return A @ x + B @ u

    plant = ode_builder(
        inputs=1,
        parameters=0,
        function=dxdt,
        initial_state=[1, 1, 1, 1]
    )

    def feedback(x, parameters):
        P = np.array(parameters).reshape(4, 4)
        return -B.T @ P @ x

    controller = function_wrapper(
        inputs=4, outputs=1, parameters=16, function=feedback, name='LQR'
    )
    components = [plant, controller]
    wires = [(controller.output, plant.input),
                   (plant.output, controller.input)]

    def biggest_eigenvalue(p):
        v = np.real(np.linalg.eigvals(p.respahe(4, 4))).max()
        return v

    constraints = [
        function(f=biggest_eigenvalue, x=controller.parameters) <= 0
    ]

    system = System(
        components=components,
        wires=wires
    )

    Q = np.eye(4)
    one_step_loss = quad_form(Q, plant.output) + sum_of_squares(controller.output)
    loss = integral(one_step_loss, 0, 10)

    problem = Problem(system, loss, constraints)

    with Timer() as timer:
        soln = problem.solve()
        duration = timer.elapsed()

    # test that we have the right solution
    epsilon = 0.0001
    P_test = solve_continuous_are(A, B, Q, np.array([[1]]))
    x = plant.outputs

    assert np.linalg.norm(P_test - loss.value) < epsilon
    assert np.linalg.norm(x(10)) < epsilon
    assert soln.duration > duration

    print(loss.value)
    raise NotImplementedError("Check the loss is within tolerance")
