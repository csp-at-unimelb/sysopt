import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite

class ConstVel2D(Block):
    # A block used for testing with constant velocity
    def __init__(self,vx=1,vy=1):
        self.vx=vx
        self.vy=vy
        metadata = Metadata(
            inputs=[],
            state=[],
            parameters=[],
            outputs=['dx','dy']
        )

        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        vx = self.vx
        vy = self.vy
        return [vx, vy]


class I2DoF(Block):
    # Translation kinetics in X and Y with simple thrust and drag
    def __init__(self,
                 x0=0,
                 y0=0,
                 vx=0,
                 vy=0):
        # TODO: What is the best structure to compile forces?
        # Should we have a separate block combine gravity, drag, thrust etc?
        metadata = Metadata(
            inputs=['Fd','Ft','g'],
            state=['x','y','dx','dy'],
            parameters=['m'],
            outputs=['x','y','dx','dy']
        )

        super().__init__(metadata)

    def compute_dynamics(self, t, state, algebraic, inputs, parameters):
        Fd, Ft, g = inputs
        m, = parameters
        x, y, dx, dy = state
        v = (dx**2 + dy**2)**0.5
        Fx = (Ft-Fd) * (dx/v)
        Fy = (Ft-Fd) * (dy/v) + g
        dynamics = [
            dx,
            dy,
            Fx/m,
            Fy/m
        ]
        return dynamics

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        x, y, dx, dy = state
        return [x, y, dx, dy]
        


