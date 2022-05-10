"""Collection of blocks needed for 3-DoF vehicle model.


"""

from sysopt import Block, Metadata
from sysopt.backends import sin, cos
import numpy as np


class VehicleParameters(Block):
    """
    Container for vehicle design parameters
    """
    def __init__(self):
        metadata = Metadata(
            parameters=["S_char in m2" ], #, "Vehicle mass (dry) in kg"],
            outputs=["S_char"] #, "Vehicle_mass_dry"],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        #S_char, vehicle_mass_dry = parameters
        S_char = parameters[0]
        return [
            S_char,
            #vehicle_mass_dry
        ]


class CombineForces3DoF(Block):
    """
    Item to combine forces for trimmed 3-DoF simulation.
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["phi", "C_D", "C_Y", "C_L", "thrust_1", "thrust_2", "thrust_3", "S_char", "Density", "Velocity"],
            outputs=["f1_total", "f2_total", "f3_total", "f1_aero", "f2_aero", "f3_aero", "thrust_1", "thrust_2", "thrust_3"],
            parameters = [],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        phi, C_D, C_Y, C_L, thrust_1, thrust_2, thrust_3, S_char, density, velocity = inputs

        qS = 0.5 * density * velocity**2 * S_char

        force_1 = - qS * C_D
        force_2 = sin(phi) * qS * C_L
        force_3 = -cos(phi) * qS * C_L

        return [
            force_1 + thrust_1,
            force_2 + thrust_2,
            force_3 + thrust_3,
            force_1,
            force_2,
            force_3,
            thrust_1,
            thrust_2,
            thrust_3
        ]
