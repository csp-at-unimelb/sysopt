"""Fundamental types and type annotations for `sysopt`."""

from dataclasses import dataclass
from numbers import Number
from typing import NewType, Iterable, Optional, Union, Callable, List
import numpy as np


Numeric = NewType('Numeric', Union[Iterable[Number], np.ndarray])

Time = NewType('Time', Number)
States = NewType('States', Optional[Numeric])
Algebraics = NewType('Algebraics', Optional[Numeric])
Inputs = NewType('Inputs', Optional[Numeric])
Parameters = NewType('Parameters', Optional[Numeric])

BlockFunction = NewType(
    'BlockFunction',
    Callable[[Time, States, Algebraics, Inputs, Parameters],
             Numeric]
)

ParameterisedConstant = NewType(
    'ParameterisedConstant', Callable[[Parameters], Numeric]
)

VectorField = NewType(
    'VectorField',
    Callable[[Time, States, Inputs, Parameters], Numeric]
)

StatelessFunction = NewType(
    'StatelessFunction',
    Callable[[Time, Inputs, Parameters], Numeric]
)


@dataclass
class Signature:
    """The dimension of the corresponding spaces."""
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
    """Descriptions, and additional information for each subspace."""

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

    @staticmethod
    def from_signature(sig):
        return Metadata(
            inputs=[f'input {i}' for i in range(sig.inputs)],
            outputs=[f'output {i}' for i in range(sig.outputs)],
            state=[f'state {i}' for i in range(sig.state)],
            constraints=[f'constraint {i}' for i in range(sig.constraints)],
            parameters=[f'parameter {i}' for i in range(sig.parameters)]
        )


@dataclass
class Domain:
    time: int = 1
    states: int = 0
    constraints: int = 0
    inputs: int = 0
    parameters: int = 0

    def __iter__(self):
        return iter((self.time, self.states, self.constraints,
                    self.inputs, self.parameters))

    def __getitem__(self, item):

        return list(self)[item]

    def __iadd__(self, other):
        self.states += other.states
        self.constraints += other.constraints
        self.inputs += other.inputs
        self.parameters += other.parameters
        return self

    def __add__(self, other):
        obj = Domain(*self)
        obj += other
        return obj

    def __eq__(self, other):
        return all(i == j for i, j in zip(self, other))
