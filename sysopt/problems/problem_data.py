"""Data structures for describing optimisation or integration problems."""

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict

import numpy as np

from sysopt.symbolic import (
    Matrix, Variable, ExpressionGraph, ConstantFunction,
    GraphWrapper,  PiecewiseConstantSignal
)

Bounds = namedtuple('Bounds', ['upper', 'lower'])


@dataclass
class Domain:
    """Domain description of sysopt common function"""
    time: int = 1
    states: int = 0
    constraints: int = 0
    inputs: int = 0
    parameters: int = 0

    _letters = {
        'time': 't',
        'states': 'x',
        'constraints': 'z',
        'inputs': 'u',
        'parameters': 'p'
    }

    def __iter__(self):
        return iter((self.time, self.states, self.constraints,
                    self.inputs, self.parameters))

    def __getitem__(self, item):
        return list(self)[item]

    def copy(self):
        return Domain(*self)

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
        try:
            return all(i == j for i, j in zip(self, other))
        except TypeError:
            return False

    @staticmethod
    def index_of_field(field_name: str):
        return ['time', 'states', 'constraints', 'inputs', 'parameters'].index(
            field_name
        )

    @staticmethod
    def letter_of_field(field_name: str):
        return Domain._letters[field_name]

    @staticmethod
    def letters():
        return Domain._letters.values()


@dataclass
class FlattenedSystem:
    """Container for flattened system functions."""
    initial_conditions: Optional[GraphWrapper] = None
    vector_field: Optional[GraphWrapper] = None
    state_transitions: Optional[Tuple[int,
                                      GraphWrapper,
                                      Optional[ExpressionGraph]]] = None
    output_map: Optional[GraphWrapper] = None
    constraints: Optional[GraphWrapper] = None
    tables: Optional[dict] = None
    domain: Domain = None
    parameter_map: Optional[GraphWrapper] = None


@dataclass
class Quadratures:
    """Container for quadratures associated with a given system."""
    output_variable: Variable
    vector_quadrature: Union[ExpressionGraph, GraphWrapper]
    regularisers: List[Variable] = field(default_factory=list)


@dataclass
class ConstrainedFunctional:
    r"""Container for a representation of a functional.

    Here
    ..math::
        value(p) := v(T, y(T), q(T), p)
        T = final_time(p)
        \dot{q} = quadratures(t, y(t), p)
        y, q, p \in constraints

    Where $y$ is generated from the flattened system:
    ..math::
        \dot{x} = f(t, x,z,p)
        y = g(t, x,z,p)
        0 = h(t,x,z,p)
        x(0) = \chi(p)

    """

    value: GraphWrapper
    """Represents a function from `(t, p, rho) -> value`
    Implicit arguments are `y(t)` and `q(t)` which are paths
    generated from solving an integral equation."""

    system: FlattenedSystem
    """System level model which produces the path `p -> y(t; p)` """

    parameters: Dict[Variable, List[int]]
    """List of the free parameters"""

    parameter_map: Union[GraphWrapper, ConstantFunction]
    """Mapping from the free parameters to the system parameters"""

    quadratures: Optional[GraphWrapper]
    """Vector-valued quadratures that are solved along side y(t);
    ie so that p-> (y(t;p), q(t;p))"""

    final_time: Union[GraphWrapper, ConstantFunction]
    """Terminal time, (interpreted as a function of p)"""

    point_constraints: List[GraphWrapper] = field(default_factory=list)
    """List of equality or inequality constraints"""
    path_constraints: List[GraphWrapper] = field(default_factory=list)


@dataclass
class SolverOptions:
    """Configuration Options for Optimisation base problems."""
    control_hertz: int = 10     # hertz
    degree: int = 3             # Collocation polynomial degree


@dataclass
class MinimumPathProblem:
    """Optimal Path Problem Specification"""
    state: Tuple[Variable, Bounds]
    control: Tuple[Variable, Bounds]
    parameters: Optional[List[Variable]]
    vector_field: ExpressionGraph
    initial_state: Union[Matrix, np.ndarray, list, ExpressionGraph]
    running_cost: Optional[ExpressionGraph]
    terminal_cost: Optional[ExpressionGraph]
    constraints: Optional[List[ExpressionGraph]] = None

    def __post_init__(self):
        if isinstance(self.state, Variable):
            bounds = Bounds([-np.inf]*len(self.state),
                            [np.inf]*len(self.state))
            self.state = (self.state, bounds)
        if isinstance(self.control, Variable):
            bounds = Bounds([-np.inf] * len(self.control),
                            [np.inf] * len(self.control))
            self.control = (self.control, bounds)


@dataclass
class CodesignSolution:
    cost: float
    argmin: Dict[Union[Variable, PiecewiseConstantSignal],
                 Union[float, np.ndarray]]
    time: np.ndarray
    outputs: np.ndarray
    quadratures: np.ndarray
    path_constraints: np.ndarray
    point_constraints: np.ndarray

