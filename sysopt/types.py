from dataclasses import dataclass
from numbers import Number
from typing import Union, NewType, List, Optional
import numpy as np
from collections import namedtuple

slice_args = namedtuple('slice_args', ['start', 'stop'])
Numeric = NewType('Numeric', Union[Number, np.ndarray])


@dataclass
class Signature:
    inputs: int = 0
    state: int = 0
    constraints: int = 0
    outputs: int = 0
    parameters: int = 0

    def __add__(self, other: 'Signature'):
        s = Signature()
        s += self
        s += other
        return s

    def __iadd__(self, other):
        self.inputs += other.inputs
        self.outputs += other.outputs
        self.state += other.state
        self.parameters += other.parameters
        self.constraints += other.constraints
        return self

    def __iter__(self):
        return iter((self.inputs, self.outputs,
                     self.state, self.constraints, self.parameters))


@dataclass
class Metadata:
    state: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    parameters: Optional[List[str]] = None

    @property
    def signature(self):
        return Signature(
            inputs=len(self.inputs) if self.inputs else 0,
            outputs=len(self.outputs) if self.outputs else 0,
            constraints=len(self.constraints) if self.constraints else 0,
            state=len(self.state) if self.state else 0,
            parameters=len(self.parameters) if self.parameters else 0
        )


def get_size(index_or_slice, max_value):
    if isinstance(index_or_slice, slice):
        start = get_size(index_or_slice.start, max_value)
        stop = get_size(index_or_slice.stop, max_value)
        step = abs(index_or_slice.step) or 1
        return abs(stop - start) // step

    if -max_value < index_or_slice < max_value:
        return 1

    raise KeyError(f"Invalid key {index_or_slice}")
