"""Collection of gravity models.

Collection of gravity models to be used by sysopt for hypersonic vehicle simulations.

"""

from sysopt import Block, Metadata
from sysopt.backends import constant as C


class GravityConstant(Block):
    """
    Gravity model with constant conditions (gx, gy, gz)
    described by parameters:
    ['gravity_x in m/s2', 'gravity_y in m/s2', 'gravity_z in m/s2']
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude"],
            outputs=["gravity_x", "gravity_y", "gravity_z"],
            parameters=["gravity_x in m/s2", "gravity_y in m/s2", "gravity_z in m/s2"]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return parameters


class GravityFlat(Block):
    """
    Gravity model using altitude
    """
    def __init__(self):
        metadata = Metadata(
            inputs=["Altitude"],
            outputs=["gravity_x", "gravity_y", "gravity_z"],
            parameters=[]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        altitude, = inputs

        R_e = 6371005  # earth radius
        GM = 3.986005e14  # GM based on 1984

        # g0 = 9.8066  # average value for g in m/s2
        s = R_e + altitude

        gravity_x = 0.
        gravity_y = 0.
        gravity_z = GM / (s*s)

        return [
            gravity_x,
            gravity_y,
            gravity_z
        ]

class GravityConstant2D(Block):
    """
    Gravity model using altitude
    """
    def __init__(self):
        metadata = Metadata(
            inputs=[],
            outputs=["gravity_x", "gravity_y"],
            parameters=[]
        )
        super().__init__(metadata)

    def compute_outputs(self, t, state, algebraics, inputs, parameters):

        gravity_y = C(-9.81)
        #  gravity_y = C(0.0)

        return [
            C(0.0),
            gravity_y,
        ]
