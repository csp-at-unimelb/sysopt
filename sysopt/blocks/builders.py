from sysopt.types import Metadata
from sysopt.block import Block


class FullStateOutput(Block):

    def __init__(self, metadata, dxdt, x0=None):
        assert not metadata.constraints, \
            f"{type(self)} must have no constraints"

        metadata = Metadata(
            inputs=metadata.inputs,
            outputs=metadata.state,
            state=metadata.state,
            parameters=metadata.parameters,
        )

        super().__init__(metadata)
        self.dxdt = dxdt
        self.x0 = x0 if x0 is not None else lambda p: [0] * len(metadata.state)

    def initial_state(self, parameters):
        return self.x0(parameters)

    def compute_dynamics(self, t, state, _, inputs, parameters):
        return self.dxdt(t, state, inputs, parameters)

    def compute_outputs(self, t, state, *args):
        return state


class InputOutput(Block):
    def __init__(self, metadata, function):
        assert not metadata.state and not metadata.constraints,\
            f"{type(self)} must not have state"

        super().__init__(metadata)
        self.output_function = function

    def compute_outputs(self, t, _1, _2, inputs, parameters):
        return self.output_function(inputs, parameters)
