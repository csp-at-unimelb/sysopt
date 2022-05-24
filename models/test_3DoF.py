
from sysopt.solver import SymbolDatabase,SolverContext
from sysopt import Block, Metadata, Composite
from sysopt.blocks import ConstantSignal

from sysopt.backends import heaviside
from sysopt.backends import constant as C
from sysopt.backends import KO_inputs

from sysopt.HSV.atmosphere import AtmosphereConstant, AtmosphereINT1962, AtmosphereUS1976Poly
from sysopt.HSV.aerodynamics import AerodynamicsConstantCoefficients
from sysopt.HSV.gravity import GravityConstant, GravityFlat
from sysopt.HSV.propulsion import PropulsionSimple3
from sysopt.HSV.inertia import InertiaMassOnly
from sysopt.HSV.vehicle_3dof import VehicleParameters, CombineForces3DoF
from sysopt.HSV.physical_environment import Environment_3DoF, Environment_3DoF_polar

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

class Vehicle_3DoF_polar(Composite):
    def __init__(self):
        self.aero = AerodynamicsConstantCoefficients()
        self.prop = PropulsionSimple3()
        #self.inertia = InertiaMassOnly()
        self.vehicle_parameters = VehicleParameters()
        self.CombineForces3DoF = CombineForces3DoF()

        components = [self.aero, self.prop, self.vehicle_parameters, self.CombineForces3DoF]

        wires = [
        ### --------- INPUTS --------------------------
        #(self.inputs[0], self.prop.inputs[0]),  # Control input - throttle [0:1]
        #(self.inputs[1], self.aero.inputs[1]),  # Control input - Angle of Attack - alpha [rad]
        #(self.inputs[2], self.CombineForces3DoF.inputs[0]),  # Control input - Bank Angle - phi [rad]
        #(self.inputs[3], self.prop.inputs[1]), # link Altitude
        #(self.inputs[4], self.CombineForces3DoF.inputs[9]),  # link velocity
        #(self.inputs[5], self.prop.inputs[2]),  # link density
        #(self.inputs[5], self.CombineForces3DoF.inputs[8]),  # link density
        #(self.inputs[6], self.aero.inputs[0]),  # link Mach
        #(self.inputs[6], self.prop.inputs[3]),  # link Mach
        # (self.inputs[7], )  # link to Temperature

        ### -------- VEHICLE PARAMETERS --------------
        (self.vehicle_parameters.outputs[0], self.CombineForces3DoF.inputs[7]),  # link S_Char
        #(self.vehicle_parameters.outputs[1], self.inertia.inputs[0]),  # vehicle_mass_dry

        ### -------- PROPULSION -----------------------
        (self.prop.outputs[0:3], self.CombineForces3DoF.inputs[4:7]), # link [t1, t2, t3]
        (self.prop.outputs[3], self.outputs[9]),  # link fuel mass
        (self.prop.outputs[4], self.outputs[10]),  # link ISP

        ### ---------AERO -----------------------------
        (self.aero.outputs[0:3], self.CombineForces3DoF.inputs[1:4]),  # link [CD, CY, C_L]

        ### ------ SUB-COMPONENT OUTPUTS TO COMPOSITE OUTPUT ------
        (self.CombineForces3DoF.outputs[0:9], self.outputs[0:9]),  # link [f1, f2, f3, f1_a, f2_a, f3_a, t1, t2, t3]
        #(self.vehicle_parameters.outputs[1], self.outputs[9]),  # link vehicle_mass_dry
        #(self.inertia.outputs[0], self.outputs[9]),  # link vehicle_mass_wet
        #(self.inertia.outputs[1], self.outputs[10]),  # link fuel mass
        (self.aero.outputs[0:3], self.outputs[11:14]), # link [CD, CY, CL]
        #(self.aero.outputs[3], self.outputs[4]),  # link to flap Angle
        #(self.prop.outputs[4], self.outputs[5]),  # link to ISP
        ]
        super().__init__(components, wires)

class ThreeDoFSimulationPolar(Composite):
    def __init__(self):
        self.gravity = GravityConstant()
        #  self.gravity = GravityFlat()
        #  self.atmos = AtmosphereConstant()
        self.atmos =  AtmosphereUS1976Poly()
        self.physics_environment = Environment_3DoF_polar()
        self.vehicle = Vehicle_3DoF_polar()
        components = [self.gravity, self.atmos, self.physics_environment, self.vehicle]
        wires = [
            #(self.inputs[0], self.vehicle.inputs[0]),  # Control input - throttle [0:1]
            #(self.inputs[1], self.vehicle.inputs[1]),  # Control input - Angle of Attack - alpha [rad]
            #(self.inputs[2], self.vehicle.inputs[2]),  # Control input - Bank Angle - phi [rad]
            #
            (self.vehicle.outputs[0:3], self.physics_environment.inputs[0:3]),  # link [f1, f2, f3]
            (self.vehicle.outputs[9], self.physics_environment.inputs[3]), # link fuel mass
            (self.vehicle.outputs[10], self.physics_environment.inputs[4]), # ISP
            (self.gravity.outputs[0:3], self.physics_environment.inputs[5:8]),  # link [g_x, g_y, g_z]
            #
            (self.physics_environment.outputs[0], self.atmos.inputs[1]),  # link velocity
            (self.physics_environment.outputs[0], self.vehicle.CombineForces3DoF.inputs[9]),  # link velocity
            #
            (self.physics_environment.outputs[1], self.atmos.inputs[0]),  # link altitude
            (self.physics_environment.outputs[1], self.gravity.inputs[0]),  # link altitude
            (self.physics_environment.outputs[1], self.vehicle.prop.inputs[1]),  # link altitude
            #(self.physics_environment.outputs[11], self.vehicle.prop.inputs[4]),  # Fuel_mass
            #(self.physics_environment.state[6], self.vehicle.prop.inputs[4]),  # Fuel_mass
            #
            (self.atmos.outputs[0], self.vehicle.prop.inputs[2]), # link to density
            (self.atmos.outputs[0], self.vehicle.CombineForces3DoF.inputs[8]), # link to density

            (self.atmos.outputs[1], self.vehicle.aero.inputs[0]), # link to Mach
            (self.atmos.outputs[1], self.vehicle.prop.inputs[3]), # link to Mach
            #
            (self.vehicle.outputs[0:14], self.outputs[0:14]), # link outputs from Vehicle
            (self.physics_environment.outputs[0:4], self.outputs[14:18]), #link outputs from physics
            #(self.vehicle.aero.outputs[0:3], self.outputs[22:25])
            #(self.gravity.outputs[0:3], self.outputs[17:20]), # link outputs from gravity
            #(self.atmos.outputs[0:3], self.outputs[20:23]) # link outputs from atmos
        ]
        super().__init__(components, wires)

class Vehicle_3DoF(Composite):
    def __init__(self):
        self.aero = AerodynamicsConstantCoefficients()
        self.prop = PropulsionSimple3()
        #self.inertia = InertiaMassOnly()
        self.vehicle_parameters = VehicleParameters()
        self.CombineForces3DoF = CombineForces3DoF()

        components = [self.aero, self.prop, self.vehicle_parameters, self.CombineForces3DoF]

        wires = [
        ### --------- INPUTS --------------------------
        #(self.inputs[0], self.prop.inputs[0]),  # Control input - throttle [0:1]
        #(self.inputs[1], self.aero.inputs[1]),  # Control input - Angle of Attack - alpha [rad]
        #(self.inputs[2], self.CombineForces3DoF.inputs[0]),  # Control input - Bank Angle - phi [rad]
        #(self.inputs[3], self.prop.inputs[1]), # link Altitude
        #(self.inputs[4], self.CombineForces3DoF.inputs[9]),  # link velocity
        #(self.inputs[5], self.prop.inputs[2]),  # link density
        #(self.inputs[5], self.CombineForces3DoF.inputs[8]),  # link density
        #(self.inputs[6], self.aero.inputs[0]),  # link Mach
        #(self.inputs[6], self.prop.inputs[3]),  # link Mach
        # (self.inputs[7], )  # link to Temperature

        ### -------- VEHICLE PARAMETERS --------------
        (self.vehicle_parameters.outputs[0], self.CombineForces3DoF.inputs[7]),  # link S_Char
        #(self.vehicle_parameters.outputs[1], self.inertia.inputs[0]),  # vehicle_mass_dry

        ### -------- PROPULSION -----------------------
        (self.prop.outputs[0:3], self.CombineForces3DoF.inputs[4:7]), # link [t1, t2, t3]
        (self.prop.outputs[3], self.outputs[9]),  # link fuel mass
        (self.prop.outputs[4], self.outputs[10]),  # link ISP

        ### ---------AERO -----------------------------
        (self.aero.outputs[0:3], self.CombineForces3DoF.inputs[1:4]),  # link [CD, CY, C_L]

        ### ------ SUB-COMPONENT OUTPUTS TO COMPOSITE OUTPUT ------
        (self.CombineForces3DoF.outputs[0:9], self.outputs[0:9]),  # link [f1, f2, f3, f1_a, f2_a, f3_a, t1, t2, t3]
        #(self.vehicle_parameters.outputs[1], self.outputs[9]),  # link vehicle_mass_dry
        #(self.inertia.outputs[0], self.outputs[9]),  # link vehicle_mass_wet
        #(self.inertia.outputs[1], self.outputs[10]),  # link fuel mass
        (self.aero.outputs[0:3], self.outputs[11:14]), # link [CD, CY, CL]
        #(self.aero.outputs[3], self.outputs[4]),  # link to flap Angle
        #(self.prop.outputs[4], self.outputs[5]),  # link to ISP
        ]
        super().__init__(components, wires)

class ThreeDoFSimulation(Composite):
    def __init__(self):
        self.gravity = GravityConstant()
        #  self.gravity = GravityFlat()
        #  self.atmos = AtmosphereConstant()
        self.atmos =  AtmosphereUS1976Poly()
        self.physics_environment = Environment_3DoF()
        self.vehicle = Vehicle_3DoF()
        components = [self.gravity, self.atmos, self.physics_environment, self.vehicle]
        wires = [
            #(self.inputs[0], self.vehicle.inputs[0]),  # Control input - throttle [0:1]
            #(self.inputs[1], self.vehicle.inputs[1]),  # Control input - Angle of Attack - alpha [rad]
            #(self.inputs[2], self.vehicle.inputs[2]),  # Control input - Bank Angle - phi [rad]
            #
            (self.vehicle.outputs[0:3], self.physics_environment.inputs[0:3]),  # link [f1, f2, f3]
            (self.vehicle.outputs[9], self.physics_environment.inputs[3]), # link fuel mass
            (self.vehicle.outputs[10], self.physics_environment.inputs[4]), # ISP
            (self.gravity.outputs[0:3], self.physics_environment.inputs[5:8]),  # link [g_x, g_y, g_z]
            #
            (self.physics_environment.outputs[0], self.atmos.inputs[1]),  # link velocity
            (self.physics_environment.outputs[0], self.vehicle.CombineForces3DoF.inputs[9]),  # link velocity
            #
            (self.physics_environment.outputs[1], self.atmos.inputs[0]),  # link altitude
            (self.physics_environment.outputs[1], self.gravity.inputs[0]),  # link altitude
            (self.physics_environment.outputs[1], self.vehicle.prop.inputs[1]),  # link altitude
            #(self.physics_environment.outputs[11], self.vehicle.prop.inputs[4]),  # Fuel_mass
            #(self.physics_environment.state[6], self.vehicle.prop.inputs[4]),  # Fuel_mass
            #
            (self.atmos.outputs[0], self.vehicle.prop.inputs[2]), # link to density
            (self.atmos.outputs[0], self.vehicle.CombineForces3DoF.inputs[8]), # link to density

            (self.atmos.outputs[1], self.vehicle.aero.inputs[0]), # link to Mach
            (self.atmos.outputs[1], self.vehicle.prop.inputs[3]), # link to Mach
            #
            (self.vehicle.outputs[0:14], self.outputs[0:14]), # link outputs from Vehicle
            (self.physics_environment.outputs[0:12], self.outputs[14:26]), #link outputs from physics
            #(self.vehicle.aero.outputs[0:3], self.outputs[22:25])
            #(self.gravity.outputs[0:3], self.outputs[17:20]), # link outputs from gravity
            #(self.atmos.outputs[0:3], self.outputs[20:23]) # link outputs from atmos
        ]
        super().__init__(components, wires)

class ConstantControl(Composite):
    def __init__(self):
        self.control_signals = ConstantSignal(outputs=3)
        self.sim = ThreeDoFSimulationPolar()
        components = [self.control_signals, self.sim]
        wires = [
            (self.control_signals.outputs[0], self.sim.vehicle.prop.inputs[0]),
            (self.control_signals.outputs[1], self.sim.vehicle.aero.inputs[1]),
            (self.control_signals.outputs[2], self.sim.vehicle.CombineForces3DoF.inputs[0]),
            (self.sim.outputs[0:26], self.outputs[0:26]),

        ]
        super().__init__(components, wires)

"""
Mass_initial_dry = 27kg
Mass_fuel = 3kg
Weight = 30*9.81 = 294.3 N

0.5*rho*V2 * Cl * S = 294.3N
Hence V = sqrt(294.3 / (0.5*1.225*0.5*1)) = 37.97m/s

Drag = 0.5*rho*V2*S* 0.01 =

"""

def simulate():
    model = ThreeDoFSimulationPolar()

    #  default_parameters = {
        #  model.control_signals.parameters[0]: 1,  # set thrust
        #  model.control_signals.parameters[1]: np.deg2rad(0.),  # set angle of attack
        #  model.control_signals.parameters[2]: np.deg2rad(2.),  # set bank angle
        #  f'{model.sim.gravity}/gravity_x in m/s2': 0.,
        #  f'{model.sim.gravity}/gravity_y in m/s2': 0.,
        #  f'{model.sim.gravity}/gravity_z in m/s2': 9.81,
        #  f'{model.sim.atmos}/density in kg/m3': 1.225,
        #  f'{model.sim.atmos}/speed of sound in m/s': 347,
        #  f'{model.sim.atmos}/temperature in K': 300,
        #  f'{model.sim.physics_environment}/X0': 0.,
        #  f'{model.sim.physics_environment}/Y0': 0.,
        #  f'{model.sim.physics_environment}/Z0': -15000.,
        #  f'{model.sim.physics_environment}/U0': 30.99,
        #  f'{model.sim.physics_environment}/V0': 0.,
        #  f'{model.sim.physics_environment}/W0': 0.,
        #  f'{model.sim.physics_environment}/Vehicle mass (dry) in kg': 27,
        #  #f'{model.sim.physics_environment}/Initial Fuel mass in kg': 3,

        #  f'{model.sim.vehicle.vehicle_parameters}/S_char in m2': 1,
        #  #f'{model.sim.vehicle.vehicle_parameters}/Vehicle mass (dry) in kg': 27,
        #  #f'{model.sim.vehicle.inertia}/Vehicle mass (dry) in kg': 27,
        #  #f'{model.sim.vehicle.inertia}/Initial Fuel mass in kg': 3,
        #  f'{model.sim.vehicle.aero}/CL_0': 0.5,
        #  f'{model.sim.vehicle.aero}/CL_a': 0.05,
        #  f'{model.sim.vehicle.aero}/CD_0': 0.05,
        #  f'{model.sim.vehicle.aero}/k': 0,
        #  #f'{model.sim.vehicle.prop}/Inital Fuel mass in kg': 3,
        #  f'{model.sim.vehicle.prop}/thrust_max in N': 29.4,
        #  f'{model.sim.vehicle.prop}/ISP in seconds': 300,
    #  }

    backend = SymbolDatabase()
    
    KO = KO_inputs(backend,model)
    KO.save("GHAME3_polar")

    breakpoint()

    return 

trajectory = simulate()



