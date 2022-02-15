from codesign import Block, Signature, Metadata
from codesign.functions import sin, cos


class HypersonicVehicleBlock(Block):
    def __init__(self):
        self.metadata = Metadata(
            inputs=["Thrust", "Lift", "Drag", " Pitching Moment"],
            state=["Height", "Velocity", "AoA", "Body Angle", "Pitch Rate"],
            outputs=["Height", "Velocity", "AoA", "Body Angle", "Pitch Rate"],
            parameters=["mass", "gravity", "Inertial moment"]
        )

        super(HypersonicVehicleBlock, self).__init__(self.metadata.signature)

    def expressions(self):
        h, V, alpha, theta, Q = self.state
        T, L, D, M = self.inputs
        m, g, I_yy = self.parameters

        return [
            V * sin(theta - alpha),
            (T * cos(alpha) - D) / m - g * sin(theta - alpha),
            (-T * sin(alpha) - L) / (m * V) + Q + (g / V) * cos(theta - alpha),
            Q,
            M / I_yy
        ]

