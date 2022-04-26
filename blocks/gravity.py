import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite

class FlatEarthGravity(Block):
    def __init__(self):
        metadata = Metadata(
            inputs=[],
            state=[],
            parameters=[],
            outputs=['g']
        )
        super().__init__(metadata)

    def compute_outputs(self,t,state,algebraic,inputs,parameters):
        g = 9.81 # m/s2
        return [g]
