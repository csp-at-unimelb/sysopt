"""Collection of blocks needed for 3-DoF vehicle model.


"""

from sysopt import Block, Metadata
import numpy as np

from sysopt.backends import heaviside

class PropulsionSimple(Block):
    """
    Propuslion system with simple scalable thrust
    ['gravity_x in m/s2', 'gravity_y in m/s2', 'gravity_z in m/s2']
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Throttle", "Altitude", "Density", "Mach", "Fuel_mass"],
            #state=["Vehicle_mass_wet"],
            outputs=["thrust_1", "thrust_2", "thrust_3", "Fuel_flow", "isp"],
            parameters=["thrust_max in N", "ISP in seconds"]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        g0=9.81
        thrust_max, isp = parameters
        throttle, _3, _4, _5, fuel_mass = inputs
        thrust = throttle * thrust_max
        #thrust = heaviside(fuel_mass) * throttle * thrust_max
        #thrust = 0

        #thrust = throttle * thrust_max
        #fuel_flow = thrust / (g0 * isp)

        """
        if fuel_mass > 0:
            thrust =  throttle * thrust_max
        else:
            thrust = 0
        """

        fuel_flow = thrust / (g0 * isp)  # change in vehilce mass will be corrected in environment.
        fuel_flow = heaviside(fuel_mass) * fuel_flow

        return [
            thrust,
            0.,
            0.,
            fuel_flow,
            isp
        ]

class PropulsionSimple2(Block):
    """
    Propuslion system with simple scalable thrust
    ['gravity_x in m/s2', 'gravity_y in m/s2', 'gravity_z in m/s2']
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Throttle", "Altitude", "Density", "Mach"],
            state=["Fuel_mass"],
            constraints=["Fuel_flow"],
            outputs=["thrust_1", "thrust_2", "thrust_3", "Fuel_mass", "isp"],
            parameters=["thrust_max in N", "ISP in seconds"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        fuel_mass_0 = 3
        return [
            fuel_mass_0
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        m_dot_fuel, = algebraics
        return [
            -m_dot_fuel
        ]

    def compute_residuals(self, t, state, algebraics, inputs, parameters):
        g0 = 9.81
        fuel_mass, = state
        m_dot_fuel, = algebraics
        throttle, _, _, _ = inputs
        thrust_max, isp = parameters
        max_fuel_flow =  thrust_max / (g0 * isp)
        return [
            m_dot_fuel - throttle * max_fuel_flow * heaviside(fuel_mass),
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        g0 = 9.81
        fuel_mass, = state
        m_dot_fuel, = algebraics
        _, isp = parameters
        thrust = m_dot_fuel * (g0*isp)
        t1 = thrust
        t2 = 0.
        t3 = 0.
        return [
            t1,
            t2,
            t3,
            fuel_mass,
            isp
        ]

class PropulsionSimple3(Block):
    """
    Propuslion system with simple scalable thrust
    Reconfigured from PropulsionSimple2 without residuals and with dynamics
    Also removes heaviside from fuel level
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Throttle", "Altitude", "Density", "Mach"],
            state=["Fuel_mass"],
            constraints=[],
            outputs=["thrust_1", "thrust_2", "thrust_3", "Fuel_mass", "isp"],
            parameters=["thrust_max in N", "ISP in seconds"]
        )
        super().__init__(metadata)

    def initial_state(self, parameters):
        fuel_mass_0 = 3
        return [
            fuel_mass_0
        ]

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        g0 = 9.81
        fuel_mass, = state
        throttle, _, _, _ = inputs
        thrust_max, isp = parameters
        max_fuel_flow =  thrust_max / (g0 * isp)
        #  m_dot_fuel = throttle * max_fuel_flow * heaviside(fuel_mass)
        m_dot_fuel = throttle * max_fuel_flow

        return [
            -m_dot_fuel
        ]

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        g0 = 9.81
        fuel_mass, = state
        thrust_max, isp = parameters
        throttle, _, _, _ = inputs
        max_fuel_flow =  thrust_max / (g0 * isp)
        thrust = throttle * max_fuel_flow * (g0*isp)
        t1 = thrust
        t2 = 0.
        t3 = 0.
        return [
            t1,
            t2,
            t3,
            fuel_mass,
            isp
        ]

class Propulsion1D(Block):
    """
    Direct relationship between throttle and thrust
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Throttle"],
            constraints=[],
            outputs=["thrust"],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        throttle, = inputs
        return [
            throttle
        ]

