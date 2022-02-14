import numpy as np
# from codesign.common import sin, cos, ode_builder, function_wrapper


def dxdt(t, X, Y, P):
    h, V, alpha, theta, Q = X
    T, L, D, M = Y
    m, g, I_yy = P

    return [
        V * sin(theta - alpha),
        (T*cos(alpha) - D) / m - g*sin(theta - alpha),
        (-T*sin(alpha) - L)/(m * V) + Q + (g/V) * cos(theta - alpha),
        Q,
        M / I_yy
    ]


def coeffs(ins, outs, params):
    alpha, deflection, v = ins
    T, L, D, M = 0, 0, 0, 0
    return T, L, D, M

s = System('HSV Design Model')

dynamics = ode_builder(function=dxdt, inputs=4, parameters=3, initial_state=[0, 0, 0, 0, 0], name='HSV Model')
coefficients = function_wrapper(function=coeffs, inputs=3, outputs=4, parameters=2)

s.components = [dynamics, coefficients]
s.wires += [
    (coefficients.output, dynamics.inputs),
    (dynamics.output[3], coefficients.inputs[0]),
    # ...
]

s.inputs = []       # list is auto-populated
s.outputs = []      # can be added to
s.constraints = []  #

problem = Problem(s)
# questions:
# - how to specify one-step loss
# - how to specify terminal loss
# -


# problem.parameters -> returns a list of parameters
# problem.


