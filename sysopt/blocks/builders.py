"""Wrapper functions for common block builders.

This module contains the classes for constructing blocks with common
functionality; for example stateless input-output maps.

See Also:
     :class:`sysopt.Block`

"""

from typing import Union, Optional

from sysopt.types import (
    Metadata, Signature, ParameterisedConstant, StatelessFunction, VectorField
)
from sysopt.block import Block


class FullStateOutput(Block):
    """Block representing a nonlinear dynamic system.

    Args:
        metadata: An instance of sysopt.Metadata or sysopt.Signature describing
            the dimensions of the input,states and parameters spaces. The
            number of algebraic constraints must be zero (which is default).
        dxdt: The parameterised function defining the block.
        x0: The initial conditions as a function of parameters, or None which
            implies x0 is zero. If `x0` is specified, it must take as an
            argument a numeric array of the same size as `metadata.parameters`
            and return a numeric array of the same size as `metadata.states`

    """
    def __init__(self,
                 metadata: Union[Metadata, Signature],
                 dxdt: VectorField,
                 x0: Optional[ParameterisedConstant] = None):
        assert not metadata.constraints, \
            f"{type(self)} must have no constraints"

        metadata = Metadata(
            inputs=metadata.inputs,
            outputs=metadata.states,
            states=metadata.states,
            parameters=metadata.parameters,
        )

        super().__init__(metadata)
        self._dxdt = dxdt
        self._x0 = x0 if x0 is not None else lambda p: [0] * len(metadata.states)

    def initial_state(self, parameters):
        return self._x0(parameters)

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        return self._dxdt(t, states, inputs, parameters)

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states


class InputOutput(Block):
    """Block representing a stateless input-output system.

    Args:
         metadata: An instance of `sysopt.Metadata` or `sysopt.Signature` that
            defines the input, output and parameter dimension of this block.
         function: A function that maps inputs and outputs to parameters.
           The input dimension should


    """

    def __init__(self,
                 metadata: Union[Metadata, Signature],
                 function: StatelessFunction):
        assert not metadata.states and not metadata.constraints,\
            f"{type(self)} must not have states"

        super().__init__(metadata)
        self._output_function = function

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return self._output_function(t, inputs, parameters)
