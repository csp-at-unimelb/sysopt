from sysopt.symbolic.symbols import Variable, scalar_shape


class PiecewiseConstantSignal(Variable):
    """
    Args:
        name:
        frequency: Update rate (in hertz)
        shape: Vector dimensions of this variable (must be of the form `(d,)`
            where `d` is the dimension.

    """
    def __init__(self, name=None, frequency=1, shape=scalar_shape):
        super().__init__(name, shape)
        self.frequency = frequency


