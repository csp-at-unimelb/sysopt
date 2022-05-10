import numpy as np
from sysopt import Block, Metadata, Composite

class CT_VL(Block):
    """ Block that converts between local (flat earth) and velocity coordinates
    """
    
    def __init__(self,
                 s0 = 0,
                 s1 = 0,
                 s2 =0):
        metadata = Metadata(
            inputs=[''],
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
