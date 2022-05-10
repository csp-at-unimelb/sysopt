"""Collection of blocks needed for 3-DoF vehicle model.


"""

from sysopt import Block, Metadata
import numpy as np


class AerodynamicsConstantCoefficients(Block):
    """
    Simple aerodynamics model with Lift and Drag coefficient defined as:
    CL = CL_0 + CL_a * alpha_deg
    CD = CD_0 + k * (CL-CL0)**2
    where CL_0, CL_a, CD_0, and k are constants.
    alpha_deg is in degrees.
    """
    def __init__(self):
        metadata = Metadata(
            parameters=["CL_0", "CL_a", "CD_0", "k"],
            inputs=["Mach", "alpha"],
            outputs=["C_D", "C_Y", "C_L", "fA"],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        CL_0, CL_a, CD_0, k = parameters
        Mach, alpha = inputs
        alpha_deg = alpha/np.pi*180.
        CL = CL_0 + CL_a * alpha_deg
        CY = 0.
        CD = CD_0 + k * (CL-CL_0)**2
        fA = 0.
        return [
            CD,
            CY,
            CL,
            fA,
        ]


class AerodynamicsConstantCoefficients2D(Block):
    def __init__(self):
        metadata = Metadata(
            inputs = ["Velocity"],
            parameters=["C_D","C_L"],
            outputs=["C_D","C_L"]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        C_D, C_L = parameters
        return [
            C_D,
            C_L
        ]
