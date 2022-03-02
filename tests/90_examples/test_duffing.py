from examples.driven_duffing_system import DuffingComponent, DuffingSystem
from sysopt import Signature


def duffing_test_values():
    delta = 3
    beta = 5
    alpha = 7
    x = 11
    dx = 13
    u = 17
    return [delta, beta, alpha], [x, dx], [u]


def test_duffing_component_numeric():

    component = DuffingComponent()

    assert len(component.inputs) == 1
    assert len(component.outputs) == 2
    assert len(component.state) == 2
    assert len(component.parameters) == 3

    test_parameters, test_state, test_control = duffing_test_values()
    delta, beta, alpha = test_parameters
    x, dx = test_state
    u, = test_control

    # duffing equation is given by
    # x' = v
    # v' = -delta * v - beta * x - alpha * x^3 + u
    expected_result = [
        dx,
        - delta * dx - beta * x - alpha * x ** 3 + u
    ]

    dX, ddX = component.compute_dynamics(0,
                                         test_state,
                                         [],
                                         test_control,
                                         test_parameters)
    assert dX == expected_result[0]
    assert ddX == expected_result[1]

    y = component.compute_outputs(0, test_state,[], [u], test_parameters)
    assert y[0] == test_state[0]
    assert y[1] == test_state[1]


def test_duffing_casadi_symbolic():
    import casadi as cs

    ps = [cs.SX.sym(f'p_i') for i in range(3)]
    xs = [cs.SX.sym('x'), cs.SX.sym('v')]
    us = [cs.SX.sym('u')]
    ts = cs.SX.sym('t')
    component = DuffingComponent()

    dX, ddX = component.compute_dynamics(ts, xs, [], us, ps)

    f = cs.Function('f', [ts, *xs, *us, *ps], [dX, ddX])

    test_parameters, test_state, test_control = duffing_test_values()
    delta, beta, alpha = test_parameters
    x, dx = test_state
    u, = test_control

    expected_result = [dx, - delta * dx - beta * x - alpha * x ** 3 + u]
    args = [0, *test_state, u, *test_parameters]

    result = f(*args)
    assert result[0] == expected_result[0]
    assert result[1] == expected_result[1]


