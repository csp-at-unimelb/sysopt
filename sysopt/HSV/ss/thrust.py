import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite

class NoThrust(Block):
    def __init__(self):
        metadata = Metadata(
            inputs=[],
            state=[],
            parameters=[],
            outputs=['Ft']
        )
        super().__init__(metadata)

    def compute_outputs(self,t,state,algebraic,inputs,parameters):
        return [0]
