"""Ballistic Test Model"""

import numpy as np
from sysopt import Block, Metadata


class ProjectileDynamics(Block):

    def __init__(self, name=None):
        metadata = Metadata(
            states=['horizontal velocity',
                    'vertical velocity',
                    'horizontal position',
                    'vertical position'],
            constraints=['angle'],
            inputs=['Drag Force', 'Effective Gravity'],
            parameters=['mass',
                        'initial horizontal velocity',
                        'initial vertical velocity'
                        'initial horizontal position',
                        'initial vertical position'],
            outputs=['horizontal velocity', 'vertical velocity',
                     'horizontal position', 'vertical position']
        )

        super().__init__(metadata, name)

    def initial_state(self, parameters):
        return parameters[1:]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        dx, dy, *_ = states
        theta, = algebraics
        drag, g = inputs
        m, *_ = parameters

        ddx = -drag * np.cos(theta)
        ddy = -m * g - drag * np.sin(theta)
        return [ddx, ddy, dx, dy]

    def compute_residuals(self, t, states, algebraics, inputs, parameters):
        dx, dy, *_ = states
        theta, = algebraics
        return theta - np.arctan2(dy, dx)

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states


def test_case_1():

    # Problem (Dynamic Optimisation):
    # Given
    #   - ballistic model
    #   - atmosphere model (constant)
    #   - aerodynamic model
    #   - gravity model (constant)

    # choose vx, vy, m
    # minimise a_1 * T_final - a_2 * m * (dx^2 + dy^2) / 2
    # s.t. vx^2 + vy^2 = V^2 (constant)
    # vx > 0
    # [dx,dy,x,y]_0 = [v_x, v_y,0, 0]
    # x_final, y_final = [x_g, 0]
    # 120 > y > 0

    pass
