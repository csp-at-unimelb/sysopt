import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite

class SimpleAtmosphere(Block):
    def __init__(self):
        metadata = Metadata(
            inputs=[],
            state=[],
            parameters=[],
            outputs=['density','viscosity']
        )
        super().__init__(metadata)

    def compute_outputs(self,t,state,algebraic,inputs,parameters):
        density = 1.225 # kg/m3
        viscocity = 1.8e-5 # Pa.s
        return [density, viscocity]

class SphereAero(Block):
    def __init__(self):
        metadata = Metadata(
            inputs=['vx','vy','density','viscosity'],
            state=[],
            parameters=['D'],
            outputs=['Fd']
        )

        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraic, inputs, parameters):
         # Placeholder simple drag
         vx, vy, density, viscocity = inputs
         D = parameters
         Cd = 0.5
         Af = 0.25*(D**2)*np.pi
         v = (vx**2 + vy**2)**0.5
         Fd = 0.5*density*(v**2)*Af*Cd
         return [Fd]
