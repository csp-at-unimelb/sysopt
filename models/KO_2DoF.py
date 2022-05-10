
from sysopt.solver import SymbolDatabase,SolverContext
from sysopt import Block, Metadata, Composite
from sysopt.blocks import ConstantSignal

from sysopt.backends import heaviside
from sysopt.backends import constant as C
from sysopt.backends import KO_inputs

from sysopt.HSV.atmosphere import *
from sysopt.HSV.aerodynamics import *
from sysopt.HSV.gravity import *
from sysopt.HSV.propulsion import *
from sysopt.HSV.inertia import *
from sysopt.HSV.vehicle_2dof import *
from sysopt.HSV.physical_environment import *

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

class Vehicle_2DoF(Composite):
    def __init__(self):
        self.aero = AerodynamicsConstantCoefficients2D()
        self.prop = Propulsion1D()
        self.vehicle_parameters = VehicleParameters()
        self.CombineForces2DoF = CombineForces2DoF()

        components = [self.CombineForces2DoF, self.aero, self.prop, self.vehicle_parameters]

        wires = [

        #  (self.inputs[0], self.prop.inputs[0]), # link [throttle]
        ### -------- VEHICLE PARAMETERS --------------
        (self.vehicle_parameters.outputs[1], self.CombineForces2DoF.inputs[5]),  # link Area

        (self.prop.outputs[0], self.CombineForces2DoF.inputs[2]), # link [thrust]

        ### ---------AERO -----------------------------
        (self.aero.outputs[0:2], self.CombineForces2DoF.inputs[0:2]),  # link [CD, C_L]

        ### ------ SUB-COMPONENT OUTPUTS TO COMPOSITE OUTPUT ------
        (self.CombineForces2DoF.outputs[0:2], self.outputs[0:2]),  # link [f1, f2]
        (self.vehicle_parameters.outputs[0], self.outputs[2]) # link [mass]

        ]
        super().__init__(components, wires)


class TwoDoFSimulation(Composite):
    def __init__(self):
        self.gravity = GravityConstant2D()
        self.atmos =  AtmosphereConstant2D()
        self.physics_environment = Environment_2DoF()
        self.vehicle = Vehicle_2DoF()
        components = [self.gravity, self.atmos, self.physics_environment, self.vehicle]
        wires = [
            # ---- VEHICLE TO PHYSICS------------
            (self.vehicle.outputs[0:2], self.physics_environment.inputs[0:2]),  # link [f1, f2]
            (self.vehicle.outputs[2], self.physics_environment.inputs[4]),  # link [mass]

            # ---- GRAVITY TO PHYSICS------------
            (self.gravity.outputs[0:2], self.physics_environment.inputs[2:4]), # link [g_x, g_y]

            # ---- ATMOS TO PHYSICS------------
            (self.atmos.outputs[0], self.vehicle.CombineForces2DoF.inputs[3]), # link to [density]

            # ---- PHYSICS TO VEHICLE ---------
            (self.physics_environment.outputs[0], self.vehicle.CombineForces2DoF.inputs[4]) # link [velocity]
        ]
        super().__init__(components, wires)

def simulate():
    model = TwoDoFSimulation()
    #  model = Vehicle_2DoF()

    backend = SymbolDatabase()
    
    KO = KO_inputs(backend,model)
    KO.save("2DoF")

    return 

simulate()



