# @nb.skip

import pytest

# @nb.code_cell_from_text
r"""
%matplotlib notebook

import os
import pathlib
import sys
path = pathlib.Path(os.curdir)
sys.path.append(str(path.absolute().parent))


import matplotlib.pyplot as plt
plt.ion()
"""
# @nb.code_cell
from sysopt import Block, Metadata, Composite
from sysopt.symbolic import Variable, Parameter
from sysopt.solver import SolverContext
import numpy as np

g = -9.81
# @nb.text_cell
r"""
# Optimising a Ballistic Rocket Model.

Given a rocket, a drag model, a basic open-loop controller and a target 
location, minimise the time-to-target.   

"""

# @nb.text_cell
r"""

Steps to solving a system optimisation problem with `sysopt`:
1. Define model subcomponents
2. Assemble composite model & connect wires
3. Define decision variables and model parameters.
4. Set up optimisation problem: cost and constraints
5. Evaluate the problem with a feasible initial guess.
6. Run the solver.   

"""

# @nb.text_cell
r"""
# Model Setup.
Let the rocket dynamics be 
$$\ddot{\mathbf{x}} = (T - D)\dot{\hat{\mathbf{x}}} + m\mathbf{g}$$
where
- $\mathbf{x} = (x,y)$ is the position in the $x$-lateral, $y$-vertical plane,
- $\dot{\hat{\mathbf{x}}}$ is the normalised velocity vector,
- $T, D$ are the thrust and drag respectively,
- $m$ is the vehicle mass,  
- $\mathbf{g} = [0, -9.81]^T$ is the gravity vector.

"""

# @nb.text_cell
r"""
In `sysopt`, this can be modelled as below
"""


# @nb.code_cell
rocket_metadata = Metadata(
    inputs=["Thrust pct", "Drag"],
    states=["x", "y", "v_x", "v_y"],
    outputs=["x", "y", "v_x", "v_y"],
    parameters=["mass in kg", "max_thrust", "dx0", "dy0", "y0"]
)


class Rocket(Block):
    def __init__(self):
        super().__init__(rocket_metadata)

    def initial_state(self, parameters):
        _1, _2, dx0, dy0, y0 = parameters
        return [0, y0, dx0, dy0]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        x, y, dx, dy = states
        mass, thrust_max, *_ = parameters
        thrust_pct, drag_coeff = inputs
        speed = (dx ** 2 + dy ** 2) ** 0.5
        force = thrust_pct * thrust_max - drag_coeff * speed
        return [
            dx,
            dy,
            dx * force / speed,
            mass * g + dy * force / speed
        ]

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states


# @nb.text_cell
r"""
# Model Setup (cont.)

Assume that the drag model is altitude and speed dependant.
We'll use a simple model with a drag coefficient of
$$ D_c(y) = D_{max} * \exp(-\rho * y) $$   
where $D_{max}, \rho$ is the drag at sea level, and the rate of decay 
with altitude respectively.   

"""


# @nb.code_cell
class DragModel(Block):
    def __init__(self):
        super().__init__(
            Metadata(inputs=['y'],
                     outputs=['D'],
                     parameters=['coeff', "exponent"]
            )
        )

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        d_max, rho = parameters
        y, = inputs
        return d_max * np.exp(- rho * y),


# @nb.text_cell
r"""
# Model Setup (cont.)

We'll assuming something like a solid-rocket motor, and ignore mass effects
for now. 
We can treat this as an open-loop controller of the form
$$
T = \begin{cases} 1 &\quad t < t_{cutoff}\\ 0 &\quad \text{otherwise}\end{cases}
$$
where $T$ is the thrust percentage, and $t_{cutoff}$ is how long we can burn for.

"""


# @nb.code_cell
class OpenLoopController(Block):
    def __init__(self):
        super().__init__(
            Metadata(outputs=['thrust_pct'],
                     parameters=['cutoff time'])
        )

    def compute_outputs(self, t, states, algebraics, inputs, parameters):

        cutoff_time, = parameters
        return np.exp(-100 * t / cutoff_time)


# @nb.text_cell
r"""
# Model Setup (cont).

Now, we can assemble the composite model.

Here, we're specifying the outputs as $(x,y,\dot{x},\dot{y}, T)$.

"""

# @nb.code_cell
class BallisticModel(Composite):
    def __init__(self):
        self.rocket = Rocket()
        self.drag = DragModel()
        self.controller = OpenLoopController()
        components = [self.rocket, self.drag, self.controller]
        wires = [
            (self.controller.outputs, self.rocket.inputs[0]),
            (self.drag.outputs, self.rocket.inputs[1]),
            (self.rocket.outputs[1], self.drag.inputs),
            (self.controller.outputs, self.outputs[4]),
            (self.rocket.outputs, self.outputs[0:4])
        ]
        super().__init__(components, wires)


# @nb.text_cell
r"""
# Problem Setup

We'll take the final time, and the thrust cutoff time as decision variables.
 
"""


# @nb.code_cell
def evaluate():
    model = BallisticModel()
    x, y, dx, dy, u = model.outputs

    t_f = Variable('final time')
    p = Parameter(model.controller, 'cutoff time')
    x_goal = 1

    parameters = {
        f'{model.rocket}/mass in kg': 1,
        f'{model.rocket}/max_thrust': 15,
        f'{model.rocket}/y0': 0,
        f'{model.drag}/coeff': 0.1,
        f'{model.drag}/exponent': 1,
        f'{model.rocket}/dx0': 0.25,
        f'{model.rocket}/dy0': 1
    }

    with SolverContext(model, t_f, parameters) as context:
        x_T = x(context.t_final)    # Final positions
        y_T = y(context.t_final)

        cost = context.t_final + y_T ** 2 + (x_T - x_goal) ** 2

        constraints = [
            context.t_final <= 10,
            0 < p,
            p <= context.t_final,
            t_f > 0
        ]

        problem = context.problem([t_f, p], cost, subject_to=constraints)

        candidate_solution = problem([1, 0.5])

    return candidate_solution


# @nb.text_cell
r"""
# Example Trajectory
"""


# @nb.code_cell_from_text
r"""

import matplotlib.pyplot as plt
import numpy as np

solution = evaluate()
T = np.linspace(0, 1, 25)
x = np.empty(shape=(5,25))
for i,t_i in enumerate(T):
    x[:, i:i+1] = solution.trajectory(t_i)[:, 0]

plt.plot(x[0,:], x[1,:])
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Trajectory with cost={solution.cost}')
plt.show()
"""


# @nb.skip
def test_ballistic_model():
    soln = evaluate()
    # assert all(c > 0 for c in soln.constraints)
    # assert 0 < soln.cost < 1e7

    parameter_sweep()

# @nb.text_cell
r"""
# Parameter Sweep

"""


# @nb.code_cell
def parameter_sweep():
    model = BallisticModel()
    x, y, dx, dy, u = model.outputs

    t_f = Variable(name='final time')
    p = Parameter(model.controller, 'cutoff time')
    x_goal = 1

    parameters = {
        f'{model.rocket}/mass in kg': 1,
        f'{model.rocket}/max_thrust': 15,
        f'{model.rocket}/y0': 0,
        f'{model.drag}/coeff': 0.1,
        f'{model.drag}/exponent': 1,
        f'{model.rocket}/dx0': 0.25,
        f'{model.rocket}/dy0': 1
    }

    n = 25
    X, Y = np.meshgrid(np.linspace(1, 2, n),
                       np.linspace(0.1, 1, n))
    Z = np.empty_like(X)

    with SolverContext(model, t_f, parameters) as context:
        x_T = x(context.t_final)
        y_T = y(context.t_final)

        cost = context.t_final + y_T ** 2 + (x_T - x_goal) ** 2

        constraints = [
            context.t_final <= 10,
            0 < p,
            p <= context.t_final,
            t_f > 0
        ]

        problem = context.problem([t_f, p], cost, subject_to=constraints)

        for i in range(n):
            for j in range(n):
                Z[i, j] = problem([float(X[i, j]), float(Y[i, j])])

    return X, Y, Z


# @nb.code_cell_from_text
r"""

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y, Z = parameter_sweep()
Z[Z > 4] = np.NaN
surf = ax.plot_surface(X, Y, Z)
ax.set_zlim(0, 2)
ax.set_xlabel('$t_f$')
ax.set_ylabel(r'$t_{cutoff}$')
ax.set_zlabel('Loss')
 
ax.set_title('Problem Surface')
plt.show()

"""