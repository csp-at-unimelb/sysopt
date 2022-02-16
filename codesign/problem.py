import casadi
import codesign as cd
import numpy as np
from codesign.core import ExpressionNode,  Parameter, Atomic, Numeric,  Signature, Ops
from codesign.block import System
from codesign.core.tree_operations import is_inequality, partition_derivatives, substitute
from codesign.core.vectors import Vector
from codesign.helpers import flatten
from typing import Optional, List, Dict, Tuple, NewType, Callable, Union, Iterable
from dataclasses import dataclass
from collections import namedtuple

Path = NewType('Path', Callable[[float], Vector])


@dataclass
class FlattenedSystem:
    X: Iterable             # Dynamics
    U: Iterable             # Inputs
    Y: Iterable             # Outputs
    P: Iterable             # Parameters
    S: Optional[Iterable]   # Slack variables
    f: Callable             # Explicit Dynamics
    g: Callable             # Algebraic constraints
    j: Callable             # Quadratures
    x0: np.ndarray          # Initial values
    t0: float
    tf: float


@dataclass
class Solution:
    value: float
    argmin: Dict[Parameter, Union[float, Path]]
    window: Tuple[float]

#
# ops = {
#     Ops.mul: lambda x, y: x * y,
#     Ops.add: lambda x, y: x + y,
#     Ops.matmul: lambda x, y: x @ y,
#     Ops.power: lambda x, y: pow(x, y),
#     Ops.sub: lambda x, y: x - y
# }
#
#
# def apply(op_str, lhs, rhs):
#     return ops[op_str](lhs, rhs)


class Minimise:
    def __init__(self,
                 objective: ExpressionNode,
                 model: System,
                 constraints: Optional[List[ExpressionNode]] = None):
        self.objective = objective
        self.model = model
        self.constraints = constraints or []

    def atoms(self):
        for equation in (self.model, self.objective, *self.constraints):
            for a in equation.atoms():
                yield a


class VariableIndexMap:
    def __init__(self, state, inputs, outputs, parameters):
        self.maps = [dict()
              for _ in (state, inputs, outputs, parameters)
         ]
        self.next_index = 0

    def push_block(self, component):
        variables = (
            component.state, component.inputs,
            component.outputs, component.parameters
        )
        component_index = self.next_index
        for v_map, var in zip(self.maps, variables):
            if var is None:
                continue
            for offset, v in enumerate(var):
                v_map.update({(component_index, offset): len(v_map)})
        self.next_index += 1


def flatten_model(model: System, symbols, t):
    # form a list of (for n nodes)
    #   X = [x_{0; 0}, ..., x_{n; k_x}]   (state)
    #   U = [u_{0; 0}, ..., u_{n; k_u}]   (inputs)
    #   Y = [y_{0; 0}, ..., y_{n; k_y}]   (output)
    #   P = [p_{0; 0}, ..., p_{n;_k_p}]   (parameters)
    #
    # split expressions into
    #    \dot{X} = F(X, U, Z, P)
    #          Y = G_{eq}(X, U, P)
    #          0 = G_{dae}(X, U, Z, P)
    #          U = LY
    #

    var_map = VariableIndexMap(*symbols)
    dxdt = []
    algebraic = []
    outputs = []

    # step one
    #
    # build tables:
    #   key | component | state | inputs | outputs | parameters
    #
    #  (for each variable class)
    #    target index | component index | offset
    #

    for i, component in enumerate(model.components):
        var_map.push_block(component)
        f, g, h = component.expressions()
        if f is not None:
            dxdt.append(f)
        if g is not None:
            outputs.append(g)
        if h is not None:
            algebraic.append(h)

    algebraic += [i - j for i, j in model.wires]

    data = FlattenedSystem(
        *symbols,
        f=vectorise(substitute(f_i, var_map) for f_i in dxdt),
        g=vectorise(substitute(g_i,  var_map) for g_i in g),
        h=vectorise(substitute(h_i, var_map) for h_i in h),
        t0=0,
        tf=1
    )


def casadi_bake(problem):
    model = problem.model
    signature = model.signature
    symbols = (
        casadi.MX.sym('X', signature.state),
        casadi.MX.sym('U', signature.inputs),
        casadi.MX.sym('Y', signature.outputs),
        casadi.MX.sym('P', signature.parameters)
    )
    t = casadi.SX.sym('t')
    substitutions, local_vars, eqns = flatten_model(model, symbols, t)
    casadi.integrator()


class Solver:
    def solve(self, problem: Minimise, window=None) -> Solution:
        window = window or [0, 1]

        casadi_bake(problem)

        return Solution(None, None, window=window)


#####
#
# To do Casadi Backend
# - For each block in the system diagram, generate casadi functions
#
#