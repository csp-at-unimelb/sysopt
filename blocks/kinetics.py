import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite
from sysopt.backend import get_default_backend

class ConstVel2D(Block):
    def __init__(self,vx=1,vy=1):
        self.vx=vx
        self.vy=vy
        metadata = Metadata(
            inputs=[],
            state=[],
            parameters=[],
            outputs=['vx','vy']
        )

        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
        vx = self.vx
        vy = self.vy
        return [vx, vy]


