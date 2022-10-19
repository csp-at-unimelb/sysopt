# @nb.code_cell_from_text
r"""
import os
import pathlib
import sys
path = pathlib.Path(os.curdir)
sys.path.append(str(path.absolute().parent))

import matplotlib.pyplot as plt
"""

# @nb.text_cell
r"""
# The Linear Quadratic Regulator

In this notebook we show how to set up a simple problem, with a linear 
plant and feedback controller.

Given an cost function, we select gains optimally using standard tools and 
then demonstrates that a parameter sweep on the cost function identifies
the locally convex region in parameter space with the optimal parameters at 
the minimum.  

"""

# @nb.code_cell
import numpy as np


# Used to solve LQR problems
from scipy.linalg import solve_continuous_are

from sysopt import Metadata
from sysopt.modelling.builders import FullStateOutput, InputOutput
from sysopt.modelling.block import Composite
from sysopt.problems import CasadiContext


# @nb.text_cell
r"""
## Modelling - Plant

Let's use the built-in helpers to construct a full state output block from a 
linear system with one input. 

We'll allow the components of the matrices to be parameters so that we can
set them later on.  

"""


# @nb.code_cell
def build_plant():
    linear_metadata = Metadata(
        states=['x_0', 'x_1'],
        inputs=['u'],
        outputs=['x_0', 'x_1'],
        parameters=['a00', 'a01', 'a10', 'a11', 'b00', 'b01']
    )

    def f(t, x, u, p):
        a00, a01, a10, a11, b00, b01 = p
        x0, x1 = x
        u0, = u
        return [
            a00 * x0 + a01 * x1 + b00 * u0,
            a10 * x0 + a11 * x1 + b01 * u0
        ]

    def x0(p):
        return [1, 1]

    plant = FullStateOutput(metadata=linear_metadata, dxdt=f, x0=x0)
    return plant


# @nb.text_cell
r"""
## Modelling - Controller

Similarly, lets use the the built-in `InputOutput` block to implement the 
feedback controller.

"""


# @nb.code_cell
def build_controller():
    controller_metadata = Metadata(
        inputs=['x0', 'x1'],
        outputs=['u'],
        parameters=['k0', 'k1']
    )

    def feedback(t, x, p):
        # u = K @ x
        k0, k1 = p
        x0, x1 = x
        return -k0 * x0 - k1 * x1

    controller = InputOutput(controller_metadata, feedback)
    return controller


# @nb.text_cell
r"""
## Modelling - Composite System

To complete the system model, instantiate the subsystems and wire them up.
 
"""


# @nb.code_cell
def build_model():
    plant = build_plant()
    controller = build_controller()
    model = Composite()
    model.components = [plant, controller]
    model.wires = [
        (plant.outputs, controller.inputs),
        (controller.outputs, plant.inputs),
        (plant.outputs, model.outputs[0:2]),
        (controller.outputs, model.outputs[2])
    ]
    return model


# @nb.text_cell
r"""
## Solving - Parameters and constants

We'll set the values of `A`, `B`, `t_final` and `Q` for our problem.
Then, map the values onto the corresponding parameters. 

"""

# @nb.code_cell
A = [[-0.1, 2],
     [-2, -1]]
B = [0, 2]
t_final = 10
Q = np.diag([1, 1, 0.1])


def build_constants(model):
    constants = {
        model.parameters[0]: A[0][0],
        model.parameters[1]: A[0][1],
        model.parameters[2]: A[1][0],
        model.parameters[3]: A[1][1],
        model.parameters[4]: B[0],
        model.parameters[5]: B[1]
    }
    return constants


# @nb.text_cell
r"""
## Solving - The uncontrolled case. 

Lets build a basic simulation and test it with no feedback.

"""


# @nb.code_cell
def simulate(p):
    model = build_model()
    constants = build_constants(model)

    with CasadiContext(model, t_final, constants) as solver:

        soln = solver.integrate(p, resolution=150)
        T = soln.t.full()
        X = soln.x.full()
    return T, X[0], X[1], X[2]

# @nb.code_cell_from_text
r"""
T, X_0, X_1, _ = simulate([0, 0])
fig, ax = plt.subplots()
ax.plot(T, X_0, 'r', label='x0')
ax.plot(T, X_1, 'g', label='x1')
ax.legend()
_ = ax.set_title('Uncontrolled Dynamics')
"""

# @nb.text_cell
r"""
## Solving - Determining the optimal gains. 

We can use `scipy`'s built in Ricatti problems to find the optimal 
gains for the infinite horizon problem given the cost function
$$ J[x,u] = \int_0^{t_final} \lVert x \rVert ^2 + \rho |u|^2 \mathrm{d}t$$
where $\rho = 0.1$ 
"""


# @nb.code_cell

A_array = np.array(A)
B_array = np.array(B).reshape(2, 1)
Q_x = Q[:2, :2]
R = Q[2:, 2:]

P = solve_continuous_are(A_array, B_array, Q_x, R)

K = np.linalg.inv(R) * B_array.T @ P
K = K.ravel()
print(K)

# @nb.text_cell
r"""
## Solving - Parameter sweep to show the cost surface 

We can set up a parameter sweep to evaluate the cost given the specified 
parameters. We'll also mark the optimal gains in red. 

It should be clear that analytic minimum lies in the middle of a local 
convex surface.  

"""


# @nb.code_cell
def do_parameter_sweep():
    model = build_model()
    constants = build_constants(model)
    n = 25
    k1_points = np.linspace(1, 4, n)
    k2_points = np.linspace(1, 4, n)
    K1, K2 = np.meshgrid(k1_points, k2_points)
    C = np.empty_like(K1)
    with CasadiContext(model, t_final, constants) as solver:
        y = model.outputs(solver.t)

        running_cost = y.T @ Q @ y
        cost = solver.integral(running_cost)

        for i in range(n):
            for j in range(n):
                params = [K1[i, j], K2[i, j]]
                C[i, j] = cost(t_final, params)

    return K1, K2, C

# @nb.code_cell_from_text
r"""
K1, K2, C = do_parameter_sweep()
c_min = C.min()
c_max = C.max()
fig, ax = plt.subplots()
contours = ax.contourf(K1, K2, C, np.linspace(c_min, c_max, 25))
fig.colorbar(contours)
ax.plot(K[0], K[1], 'ro')
_ = ax.set_title('Cost Contours')
_ = ax.set_xlabel(r'$k_1$')
_ = ax.set_ylabel(r'$k_2$')
"""

# @nb.text_cell
r"""
## Solving - Evaluating the system along the optimal path. 

We can see the optimal cost, control effort and state over time.

"""


# @nb.code_cell
def simulate_with_quadratic_cost(parameters):
    model = build_model()
    constants = build_constants(model)
    with CasadiContext(model, t_final, constants) as solver:
        y = model.outputs(solver.t)

        running_cost = y.T @ Q @ y
        cost = solver.integral(running_cost)

        y, q = solver.integrate(parameters, resolution=150)

        t_grid = q.t.full()[0]
        q_grid = q.x.full()[cost.index]
        x_grid = y.x.full()
    return t_grid, x_grid, q_grid

# @nb.code_cell_from_text
r"""
T, X, Q = simulate_with_quadratic_cost(K) 


fig, ax = plt.subplots()
ax.plot(T, Q,'k--', label='cost')
ax.plot(T, X[0],'r', label='x0')
ax.plot(T, X[1], 'g', label='x1')
ax.plot(T, X[2], 'b:', label='u')
ax.legend()
_ = ax.set_title('Infinite Horizon LQR Controller')

"""


# @nb.skip
def test_lqr_functions():
    uncontrolled = [0, 0]
    t, x0, x1, u = simulate(uncontrolled)
    assert abs(x0[-1]) < 1e-2

    controlled = K
    _ = simulate(controlled)
    _ = simulate_with_quadratic_cost(controlled)
    # Skip, since it's slow
    # _ = do_parameter_sweep()
