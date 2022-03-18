from sysopt import Block, Metadata, Composite
from sysopt.symbolic import heaviside
from sysopt.optimisation import DecisionVariable
from sysopt.solver import SolverContext

import numpy as np
g = 9.81
# Mocks


class Rocket(Block):
    def __init__(self):
        metadata = Metadata(
            inputs=["Thrust pct", "Drag"],
            state=["x", "y", "v_x", "v_y"],
            outputs=["x", "y", "v_x", "v_y"],
            parameters=["mass in kg", "max_thrust", "dx0", "dy0", "y0"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        _1, _2, dx0, dy0, y0 = parameters
        return [0, y0, dx0, dy0]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        x, y, dx, dy = state
        mass, thrust_max, *_ = parameters
        thrust_pct, drag = inputs
        force = thrust_pct * thrust_max - drag
        speed = (dx ** 2 + dy**2) ** 0.5
        return [
            dx,
            dy,
            - dx * force / speed,
            - mass * g - dy * force / speed
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return state,


class DragModel(Block):
    def __init__(self):
        super().__init__(
            Metadata(inputs=['y'], outputs=['D'],
                     parameters=['coeff', "exponent"]
            )
        )

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        d_max, rho = parameters
        y = inputs
        return d_max * np.exp(- rho * y),


class OpenLoopController(Block):
    def __init__(self):
        super().__init__(
            Metadata(
                outputs=['thrust_pct'],
                parameters=['cutoff time']
            )
        )

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        cutoff_time, = parameters
        return heaviside(cutoff_time - t),


class BallisticModel(Composite):
    def __init__(self):
        rocket = Rocket()
        drag = DragModel()
        self.controller = OpenLoopController()
        components = [rocket, drag, self.controller]
        wires = [
            (self.controller.outputs, rocket.inputs[0]),
            (drag.outputs, rocket.inputs[1]),
            (rocket.outputs[1], drag.inputs),
            (self.controller.outputs, self.outputs[4]),
            (rocket.outputs, self.outputs[0:4])
        ]
        super().__init__(components, wires)


def test_ballistic_model():

    model = BallisticModel()
    x, y, dx, dy, u = model.outputs

    t_f = DecisionVariable()
    p = DecisionVariable(model.controller, 'cutoff time')
    x_goal = 10
    y_max = 120

    parameters = {
        f'{model.name}/Rocket_1/mass in kg': 1,
        f'{model.name}/Rocket_1/max_thrust': 5,
        f'{model.name}/Rocket_1/dx0': 0,
        f'{model.name}/Rocket_1/dy0': 1,
        f'{model.name}/Rocket_1/y0': 1,
        f'{model.name}/DragModel_1/coeff': 1,
        f'{model.name}/DragModel_1/exponent': 1
    }

    with SolverContext(model, t_f, parameters) as context:
        y_T = y(context.end)
        x_T = x(context.end)

        # running_cost = 1
        # final_cost = y ** 2 +  (x - x_goal)**2

        cost = context.end + y_T ** 2 + (x_T - x_goal)**2

        constraints = [
            0 <= y(context.t),
            y(context.t) <= y_max,
            context.end <= 360,
            0 < p,
            p <= context.end,
            t_f > 0
        ]

        problem = context.minimise(cost, subject_to=constraints)
        viable_cost = problem(1, 1)

        assert viable_cost < np.inf
