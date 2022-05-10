"""Collection of gravity models.

Collection of gravity models to be used by sysopt for hypersonic vehicle simulations.

"""

from sysopt import Block, Metadata
import numpy as np
from sysopt.backends import atan2, sin, cos
# from sysopt.backends import heaviside

class Environment_3DoF(Block):
    """
    Phyisical Simualtion Environment for 3 Degree of Freedom (3-DoF) trimmed
    simulations.
    """
    def __init__(self):
        metadata = Metadata(
            state = ["X", "Y", "Z", "U", "V", "W"],
            inputs=["f1", "f2", "f3", "Fuel_Mass", "ISP", "g_x", "g_y", "g_z"],
            outputs=["Velocity", "Altitude", "Chi", "Gamma"],
            parameters=["X0", "Y0", "Z0", "U0", "V0", "W0", "Vehicle mass (dry) in kg"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        X0, Y0, Z0, U0, V0, W0, _1 = parameters
        return [
            X0,
            Y0,
            Z0,
            U0,
            V0,
            W0
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        X, Y, Z, U, V, W = state
        f1, f2, f3, fuel_mass, _1, g_x, g_y, g_z = inputs
        X0, Y0, Z0, U0, V0, W0, vehicle_mass_dry = parameters

        chi = atan2(V, U)
        gamma = atan2(-W, (U**2+V**2)**0.5)

        F1 = f1 * cos(gamma) * cos(chi)  + f2 * -sin(chi)  + f3 * sin(gamma) * cos(chi)
        F2 = f1 * cos(gamma) * sin(chi) + f2 * cos(chi)  + f3 * sin(gamma) * sin(chi)
        F3 = f1 * -sin(gamma) + f2 * 0. + f3 * cos(gamma)

        #V_ground = (U**2+V**2)**0.5
        #Vel = (U**2+V**2+W**2)**0.5
        #F1 = f1 * V_ground/Vel * U/V_ground + f2 * -V/V_ground  + f3 * W/Vel * U/V_ground
        #F2 = f1 * V_ground/Vel * V/V_ground + f2 * U/V_ground + f3 * W/Vel * V/V_ground
        #F3 = f1 * -W/Vel + f2 * 0. + f3 * V_ground/Vel

        #F1 = 0
        #F2 = 0.1
        #F3 = 0
        #Vehicle_mass_wet = 30 - t*0.01
        vehicle_mass_wet = vehicle_mass_dry + fuel_mass

        U_dot = F1/vehicle_mass_wet + g_x
        V_dot = F2/vehicle_mass_wet + g_y
        W_dot = F3/vehicle_mass_wet + g_z

        #m_dot = heaviside(vehicle_mass_wet - vehicle_mass_dry) * -1 * fuel_flow
        #m_dot = -1 * fuel_flow

        return [
            U,
            V,
            W,
            U_dot,
            V_dot,
            W_dot
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        X, Y, Z, U, V, W = state
        f1, f2, f3, fuel_mass, _1, g_x, g_y, g_z = inputs
        X0, Y0, Z0, U0, V0, W0, vehicle_mass_dry = parameters
        vehicle_mass_wet = vehicle_mass_dry + fuel_mass
        velocity = (U**2 + V**2 + W**2)**0.5
        V_ground = (U**2 + V**2)

        chi = atan2(V, U)
        gamma = atan2(-W, V_ground)

        return [
            velocity,  # 0
            -Z,  # 1
            chi,  # 2
            gamma,  # 3
            X,  # 4
            Y,  # 5
            Z,  # 6
            U,  # 7
            V,  # 8
            W,  # 9
            vehicle_mass_wet,  # 10
            fuel_mass  # 11
        ]

class Environment_2DoF(Block):
    """
    Phyisical Simualtion Environment for 3 Degree of Freedom (3-DoF) trimmed
    simulations.
    """
    def __init__(self):
        metadata = Metadata(
            state = ["X", "Y", "U", "V"],
            inputs=["f1", "f2", "g_x", "g_y","Vehicle mass (kg)"],
            outputs=["Velocity"],
            parameters=["X0", "Y0", "U0", "V0"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        X0, Y0, U0, V0 = parameters
        return [
            X0,
            Y0,
            U0,
            V0
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        X, Y, U, V = state
        f1, f2, g_x, g_y, mass = inputs

        gamma = atan2(V, U)

        F1 = f1 * cos(gamma)  - f2 * sin(gamma) 
        F2 = f1 * sin(gamma) + f2 * cos(gamma) 

        U_dot = F1/mass + g_x
        V_dot = F2/mass + g_y

        return [
            U,
            V,
            U_dot,
            V_dot
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        X, Y, U, V = state
        velocity = (U**2 + V**2)**0.5

        return [
            velocity
        ]
