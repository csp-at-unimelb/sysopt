import numpy as np
from sysopt.backends.sympy import SymbolicVector
from sysopt.solver import SymbolDatabase
import aero
import gravity
import kinetics 
import thrust

from sysopt import Block, Metadata, Composite

class Projectile2D(Composite):
    def __init__(self):
        self.atmos = aero.SimpleAtmosphere()
        self.aero = aero.SphereAero()
        self.vel = kinetics.ConstVel2D() 

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
    ctx = SymbolDatabase()
    projectile = Projectile2D()
    model = ctx.get_flattened_system(projectile)
    breakpoint()

simulate()
