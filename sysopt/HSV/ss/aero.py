import numpy as np
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

class PosCart(Block):
    """ Block to handle dynamics between position and velocity in arbitrary
    3-dimensional cartesion coordinates
    """
    
    def __init__(self,
                 s0 = 0,
                 s1 = 0,
                 s2 =0):
        metadata = Metadata(
            inputs=['v0','v1','v2'],
            state=['s0','s1','s2'],
            parameters=[],
            outputs=['s0','s1','s2']
        )
        super().__init__(metadata)

    def compute_dynamics(self, t, state, algebraic, inputs, parameters):
        v0, v1, v2 = inputs
        s0, s1, s2 = state

        dynamics = [
            v0,
            v1,
            v2,
        ]
        return dynamics

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        s0, s1, s2 = state
        return [s0, s1, s2]

class VelCart(Block):
    """ Block to handle Newtonian dynamics in abritrary -dimensional 
    cartesion coordinates, assuming constant mass
    """

    def __init__(self,
                 v0 = 0,
                 v1 = 0,
                 v2 =0):
        metadata = Metadata(
            inputs=['F0','F1','F2'],
            state=['v0','v1','v2'],
            parameters=['m'],
            outputs=['v0','v1','v2']
        )
        super().__init__(metadata)

    def compute_dynamics(self, t, state, algebraic, inputs, parameters):
        F0, F1, F2 = inputs
        v0, v1, v2 = state

        dynamics = [
            F0/m,
            F1/m,
            F2/m,
        ]
        return dynamics

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        v0, v1, v2 = state
        return [v0, v1, v2]



