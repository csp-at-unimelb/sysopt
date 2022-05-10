"""Collection of blocks needed for 3-DoF vehicle model.


"""

from sysopt import Block, Metadata
from sysopt.backends import heaviside
import numpy as np


class InertiaMassOnly(Block):
    """
    Simple inertia model that evaluates 'vehicle mass (wet)'
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Fuel_flow"],
            state=["Vehicle_mass_wet"],
            outputs=["Vehicle_mass_wet", "Fuel_mass"],
            parameters=["Vehicle mass (dry) in kg", "Initial Fuel mass in kg"],
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        Vehicle_mass_dry, fuel_mass_0 = parameters
        return [Vehicle_mass_dry + fuel_mass_0]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        vehicle_mass_wet = state[0]
        vehicle_mass_dry, _1 = parameters
        #fuel_flow = inputs[0]
        fuel_flow = 0
        mdot = 0 #heaviside(vehicle_mass_wet - vehicle_mass_dry) * -1 * fuel_flow
        return [
            mdot
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        vehicle_mass_wet = state[0]
        vehicle_mass_dry, _1 = parameters
        return [
            30, #vehicle_mass_wet,
            0, #vehicle_mass_wet - vehicle_mass_dry
        ]
