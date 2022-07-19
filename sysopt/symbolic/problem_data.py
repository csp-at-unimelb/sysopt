from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

import numpy as np

from sysopt.types import Domain
from sysopt.symbolic.symbols import (
    Matrix, Variable, ExpressionGraph, Parameter
)


Bounds = namedtuple('Bounds', ['upper', 'lower'])


@dataclass
class FlattenedSystem:
    """Container for flattened system functions."""
    initial_conditions: Optional[ExpressionGraph] = None
    vector_field: Optional[ExpressionGraph] = None
    state_transitions: Optional[Tuple[int,
                                      ExpressionGraph,
                                      Optional[ExpressionGraph]]] = None
    output_map: Optional[ExpressionGraph] = None
    constraints: Optional[ExpressionGraph] = None
    tables: Optional[dict] = None
    domain: Domain = None
    parameter_map: Optional[ExpressionGraph] = None


@dataclass
class Quadratures:
    """Container for quadratures associated with a given system."""
    output_variable: Variable
    vector_quadrature: ExpressionGraph
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

    value: ExpressionGraph
    """Represents a function from `(t, p, rho) -> value`
    Implicit arguments are `y(t)` and `q(t)` which are paths
    generated from solving an integral equation."""

    system: FlattenedSystem
    """System level model which produces the path `p -> y(t; p)` """

    parameters: List[str]
    """Names of each of the free parameters"""

    parameter_map: ExpressionGraph
    """Mapping from the free parameters to the system parameters"""

    quadratures: ExpressionGraph
    """Vector-valued quadratures that are sovled along side y(t);
    ie so that p-> (y(t;p), q(t;p))"""

    final_time: Union[ExpressionGraph, float]
    """Terminal time, (interpreted as a function of p)"""

    constraints: List[ExpressionGraph] = field(default_factory=list)
    """List of equality or inequality constraints"""

    regularisers: List[Variable] = field(default_factory=list)
    """List of regularisation variables s.t. $rho-> 0$ is the 'true' 
    function, which may not be smooth"""


@dataclass
class SolverOptions:
    """Configuration Options for Optimisation base solver."""
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
        if isinstance(self.state, (Variable, Parameter)):
            bounds = Bounds([-np.inf]*len(self.state),
                            [np.inf]*len(self.state))
            self.state = (self.state, bounds)
        if isinstance(self.control, (Variable, Parameter)):
            bounds = Bounds([-np.inf] * len(self.control),
                            [np.inf] * len(self.control))
            self.control = (self.control, bounds)
