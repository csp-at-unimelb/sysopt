"""Collection of atmospheric models.

Collection of atmospheric models to be used by sysopt for hypersonic vehicle simulations.

"""

from sysopt import Block, Metadata
from sysopt.backends import heaviside
from sysopt.backends import exp
from sysopt.backends import piecewise
import numpy as np


class AtmosphereConstant2D(Block):
    """
    Atmosphere with constant conditions (density, speed of sound, temperature)
    described by parameters:
    ['density in kg/m3', 'speed of sound in m/s', 'temperature in K']
    """
    def __init__(self):
        metadata = Metadata(
            inputs=[],
            outputs=["Density"],
            parameters=["density in kg/m3"]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        density, = parameters
        return [
            density
        ]

class AtmosphereConstant(Block):
    """
    Atmosphere with constant conditions (density, speed of sound, temperature)
    described by parameters:
    ['density in kg/m3', 'speed of sound in m/s', 'temperature in K']
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude", "Velocity"],
            outputs=["Density", "Mach", "Temperature"],
            parameters=["density in kg/m3", "speed of sound in m/s", "temperature in K"]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        density, speed_of_sound, temperature = parameters
        altitude, velocity = inputs
        return [
            density,
            velocity/speed_of_sound,
            temperature
        ]


class AtmosphereINT1962(Block):
    """
    Get atmospheric data from International Stamdard Atmosphere 1962

    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude", "Velocity"],
            outputs=["Density", "Mach", "Temperature"],
            parameters = []
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        # T0 = 288.15  # K
        # P0 = 101325  # Pa
        gamma = 1.4  # -
        R = 287.053  # J/kg-K
        R_e = 6356766. # m radius of earth

        geometric_altitude, velocity = inputs
        geopotential_altitude = R_e*geometric_altitude / (R_e+geometric_altitude)
        h = geopotential_altitude/1000  # geopotential altitude in km

        #  print("updated")
        #  T = (288.15 - 6.5*h) * heaviside(11-h) + 216.65 * heaviside(h-11)
        #  P = (101325.0 * (T/288.15)**5.2559) * heaviside(11-h) + (22632.06 * exp(-0.15769 * (h - 11))) * heaviside(h-11)

        T = piecewise( (288.15 - 6.5*h,h<=11),(216.65,True) )
        P = piecewise( (101325*((288.15 - 6.5*h)/288.15)**5.2559,h<=11),(22630*exp(-0.15769 * (h-11)),True) )

        #T = 216.65 + ((11-h) * 6.5 * heaviside(11-h))
        #P = (101325.0 * (T/288.15)**5.2559) #* heaviside(11-h)

        '''
        T = (288.15 - 6.5*h) * heaviside(11-h) + \
            216.65 * heaviside(h-11)
        P = (101325.0 * (T/288.15)**5.2559) * heaviside(11-h) + \
            (22632.06 * exp(-0.15769 * (h - 11))) * heaviside(h-11)
        '''
        """
        if 0 < h and h <= 11:
            T = 288.15 - 6.5*h
            P = 101325.0 * (T/288.15)**5.2559
        elif 11 < h and h <= 86:
            T = 216.65
            P = 22632.06 * np.exp(-0.15769 * (h - 11))
        else:
            raise Exception("'geometric altitude' = {} is out of range for atmosphere model.")
        """

        density = P / (R*T)
        speed_of_sound = (gamma * R * T)**0.5

        return [
            density,
            velocity/speed_of_sound,
            T
        ]


class AtmosphereUS1976(Block):
    """
    Get atmospheric data from US 1976 Stamdard Atmosphere
    Input is Geopotential altitude.
    Source: http://www.braeunig.us/space/atmmodel.htm#modeling
    Currently only implemented up to an altitude of 86km.
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude", "Velocity"],
            outputs=["Density", "Mach", "Temperature"],
            parameters=[],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        # T0 = 288.15  # K
        # P0 = 101325  # Pa
        gamma = 1.4  # -
        R = 287.053  # J/kg-K

        geometric_altitude, velocity = inputs
        geopotential_altitude = R_e*geometric_altitude / (R_e+geometric_altitude)
        h = geopotential_altitude/1000  # geopotential altitude in km

        if h < 0:
            T = 288.15
            P = 101325.0
        elif 0 <= h and h  <= 11:
            T = 288.15 - 6.5*h
            P = 101325.0 * (288.15 / (288.15 - 6.5 * h))**(34.1632/-6.5)
        elif 11 < h and h  <= 20:
            T = 216.65
            P = 22632.06  * np.exp(-34.1632 * (h - 11) / 216.65)
        elif 20 < h and h <= 32:
            T = 216.65 + h
            P = 5474.889 * (216.65 / (216.65 + (h - 20)))**(34.1632)
        elif 32 < h and h <= 47:
            T = 139.05 + 2.8 * h
            P = 868.0187 * (228.65 / (228.65 + 2.8 * (h - 32)))**(34.1632 / 2.8)
        elif 47 < h and h <= 51:
            T = 270.65
            P = 110.9063 * np.exp(-34.1632 * (h - 47) / 270.65)
        elif 51 < h and h <= 71:
            T = 413.45 - 2.8 * h
            P = 66.93887 * (270.65 / (270.65 - 2.8 * (h - 51)))**(34.1632 / -2.8)
        elif 71 < h and h < 84.852:
            T = 356.65 - 2.0 * h
            P = 3.956420 * (214.65 / (214.65 - 2 * (h - 71)))**(34.1632 / -2)
        else:
            raise Exception("'geometric altitude' = {} is out of range for atmosphere model.")

        density = P / (R*T)
        speed_of_sound = (gamma * R * T)**0.5

        return [
            density,
            velocity/speed_of_sound,
            T
        ]

class AtmosphereUS1976Poly(Block):
    """
    Curve-fit using US 1976 points:
    https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere#cite_note-USA_1962-4
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude", "Velocity"],
            outputs=["Density", "Mach", "Temperature"],
            parameters=[],
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        # T0 = 288.15  # K
        # P0 = 101325  # Pa
        gamma = 1.4  # -
        R = 287.053  # J/kg-K

        geometric_altitude, velocity = inputs
        #  geopotential_altitude = R_e*geometric_altitude / (R_e+geometric_altitude)
        h = -geometric_altitude/1000  # geopotential altitude in km

        C_T = [2.49698758e-05, -6.30596313e-03,  4.57479877e-01, -1.08795947e+01, 2.95390111e+02]
        C_p = [ 3.01088666e-02, -5.52942148e+00,  3.65461083e+02, -1.02955447e+04, 1.05136679e+05]

        T = C_T[0]*h**4 + C_T[1]*h**3 + C_T[2]*h**2 + C_T[3]*h + C_T[4]
        P = C_p[0]*h**4 + C_p[1]*h**3 + C_p[2]*h**2 + C_p[3]*h + C_p[4]

        density = P / (R*T)
        speed_of_sound = (gamma * R * T)**0.5

        return [
            density,
            velocity/speed_of_sound,
            T
        ]


