#!/usr/bin/env python3

r"""Goal - Fit a van der pol oscillator to a cosine wave.

A driven Van der Pol oscillation is a nonlinear system of the form

math::
    \dot{x}  = \tau x
    \dot{v} = \mu (1 - x^2)v - x - u

Here :math:`x, v, u` is the position, velocity and input respectively.
We assume :math:`u` is the output of a digital device, and hence has
values of -1, 1.

Our goal is to find :math:`\mu \tau' and a piecewise constant signal
:math:`u = \pm 1`, as well as :math:`x(0), v(0)` which given a fixed
:math:`\omega` minimises
math::

    J[x, u] = \int_0^{2\pi} |\cos(\omega t) - x(t)|^2 \df{t}

Subject to:
- Box constraints on the parameters
- Control is digital
- Periodic terminal constraints

"""
import numpy as np

from sysopt import (
    FullStateOutput, Metadata, Composite, Parameter, PiecewiseConstantSignal,
    SolverContext
)
from sysopt.blocks import ConstantSignal
import matplotlib.pyplot as plt
from sysopt.backends import scalar_ops


def dxdt(t, x, u, p):
    return [p[1]*x[1], p[0]*(1 - x[0]**2)*x[1] - x[0] - u[0]]


def x0(p):
    return [p[2], p[3]]


vdp_metadata = Metadata(
    inputs=["u"],
    states=["v'", "x"],
    parameters=["mu", "tau", "v(0)", "x(0)"]
)

vanderpol_osc = FullStateOutput(metadata=vdp_metadata, dxdt=dxdt, x0=x0)
controller = ConstantSignal(['u'])

model = Composite(components=[vanderpol_osc, controller])
model.declare_outputs(["x", "v", "u"])
model.wires = [
    (vanderpol_osc.outputs[1],  model.outputs[0]),
    (vanderpol_osc.outputs[0],  model.outputs[1]),
    (controller.outputs[0],     vanderpol_osc.inputs[0]),
    (controller.outputs[0],     model.outputs[2])
]

mu = Parameter(vanderpol_osc, 0)
tau = Parameter(vanderpol_osc, 1)
v0 = Parameter(vanderpol_osc, 2)
x0 = Parameter(vanderpol_osc, 3)
u = PiecewiseConstantSignal(['u'])

gain = 4

freq = 0.5  # Hz
eps = 0.001
t_f = 1/freq

with SolverContext(model=model, t_final=2 * t_f, constants={}) as solver:
    t = solver.t
    x_t = model.outputs(t)[0]
    u_t = model.outputs(t)[2]
    x_f = model.outputs(solver.t_final)
    cost = solver.integral((np.cos(2 * np.pi * freq * t) - gain * x_t)**2)

    constraints = [-1 <= v0, v0 <= 1,
                   -1 <= x0, x0 <= 1,
                   4 < mu,  mu < 10,
                   1 < tau, tau < 1.2,
                   -eps < x_f[0] - x0, x_f[0] - x0 < eps,
                   -eps < x_f[1] - v0, x_f[1] - v0 < eps,
                   u_t ** 2 >= 1 - eps,
                   u_t ** 2 <= 1 + eps]

    problem = solver.problem(
        arguments=[mu, tau, v0, x0, u],
        cost=cost,
        subject_to=constraints
    )

    soln = problem.solve(guess=[1, 1, 0, 0, 0])

T = soln.time.ravel()
plt.plot(T, gain * soln.outputs[0, :], label='VDP')
plt.plot(T, soln.outputs[2, :], label='u')
plt.plot(T, np.cos(2*np.pi * freq * T), label='cos')
mu_star = float(soln.argmin[0])
tau_star = float(soln.argmin[1])
plt.title("Fitting a PWM driven Van der Pol Oscillator to a sine wave")
plt.legend()
plt.show()

print( "=================================")
print( "==========Solution===============")
print(f"Optimal at (mu, tau) = %3.2f, %3.2f".format(mu_star, tau_star))

