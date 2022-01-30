import numpy as np


class Block:
    def __init__(self, name):
        if name is None:
            name = f"{self.__class__.__name__}_{self.__class__._n}"
            self.___class__._n += 1
        self.name = name

    def _call(self, *args):
        raise NotImplemented

    def input_shape(self):
        raise NotImplemented

    def parameter_shape(self):
        raise NotImplemented

    def output_shape(self):
        raise NotImplemented

    def state_shape(self):
        raise NotImplemented


class Function(Block):
    def state_shape(self):
        return 0


class Variable:
    def __init__(self, name, initial_value, bounds=None):
        pass


def shape_of(x):
    if isinstance(x, np.ndarray):
        return x.shape
    raise NotImplemented


class OdeBlock(Block):
    def __init__(self, name, f,
                 u_dim,
                 parameters=None,
                 x0=None,
                 *args, **kwargs):
        super(OdeBlock).__init__(name)
        self._f = f
        self._params = parameters
        self._initial_state = x0
        self._input_shape = u_dim

    def _call(self, ):

    def input_shape(self):
        return self._input_shape,

    def output_shape(self):
        return shape_of(self._initial_state)

    def state_shape(self):
        return shape_of(self._initial_state)

    def parameter_shape(self):
        if isinstance(self._params, dict):
            count = len([p for p in self._params.values() if isinstance(p, Variable)])
        else:
            count = len([p for p in self._params if isinstance(p, Variable)])

        count += len([v for v in self._initial_state if isinstance(v, Variable)])
        return count

