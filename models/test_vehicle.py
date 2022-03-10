import numpy as np
import casadi as cs

from sysopt import Block, Metadata, Composite

class Projectile2D(Composite):
    def __init__(self):
        self.atmos = SimpleAtmosphere()
        self.aero = SphereAero()
        self.vel = ConstantSpeed() 

        wires = [
            (self.atmos.outputs[0], self.aero.inputs[2]),
            (self.atmos.outputs[1], self.aero.inputs[3]),
            (self.vel.outputs[0], self.aero.inputs[0]),
            (self.vel.outputs[1], self.aero.inputs[1])
        ]
    
        components = [self.atmos, self.aero, self.vel]

        super().__init__(
            components=components,
            wires=wires
        )


def simulate():
    projectile = Projectile2D()
    backend = get_default_backend()
    model = backend.get_flattened_system(projectile)
    breakpoint()
