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
            parameters=["Vehicle mass (kg)","Frontal area (m2)"],
            outputs=["Vehicle mass (kg)","Frontal area (m2)"] 
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        mass,area = parameters
        return [
            mass,
            area
        ]


class CombineForces2DoF(Block):
    """
    Item to combine forces for trimmed 3-DoF simulation.
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["C_D", "C_L", "thrust", "Density", "Velocity","Area"],
            outputs=["f1_total", "f2_total"],
            parameters = [],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        C_D, C_L, thrust, density, velocity, area = inputs

        qS = 0.5 * density * velocity**2 * area

        force_1 = - qS * C_D + thrust
        force_2 = qS * C_L

        return [
            force_1,
            force_2 
        ]
