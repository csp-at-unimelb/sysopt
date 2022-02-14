import casadi
import codesign as cd
import numpy as np
from codesign.core import ExpressionNode,  Parameter, Atomic, Numeric,  Signature, Ops
from codesign.block import System
from codesign.core.tree_operations import is_inequality, partition_derivatives, substitute
from codesign.core.vectors import Vector
from typing import Optional, List, Dict, Tuple, NewType, Callable, Union
from dataclasses import dataclass

Path = NewType('Path', Callable[[float], Vector])


@dataclass
class CasadiSystemData:
    X: casadi.MX        # Dyanmics
    U: casadi.MX        # Inputs
    Y: casadi.MX        # Outputs
    P: casadi.MX        # Parameters
    S: casadi.MX        # Slack variables
    f: casadi.Function  # Explicit Dynamics
    g: casadi.Function  # Algebraic constraints
    j: casadi.Function  # Quadratures
    x0: np.ndarray      # Initial values
    t0: float
    tf: float


@dataclass
class Solution:
    value: float
    argmin: Dict[Parameter, Union[float, Path]]
    window: Tuple[float]


ops = {
    Ops.mul: lambda x, y: x * y,
    Ops.add: lambda x, y: x + y,
    Ops.matmul: lambda x, y: x @ y,
    Ops.power: lambda x, y: pow(x, y)
}


def apply(op_str, lhs, rhs):
    return ops[op_str](lhs, rhs)


def bake_tree(tree: ExpressionNode, substitutions: Dict[Atomic, Numeric]):
    try:
        children = (tree.lhs, tree.rhs)
    except AttributeError:
        return bake_atom(tree, substitutions)

    return apply(tree.op, *children)


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


def flatten(the_list):
    return [i for sublist in the_list for i in sublist]


def find_index_of(x, lookup) -> Union[int, slice]:
    # cases:
    # - x is a Variable, so we need to find and index
    # - x is a contiguous vector, so we need to find a slice
    n = 0
    print(f"Looking for {x} in {lookup}")
    if isinstance(x, Vector):
        for entry, item in lookup:
            if x is entry:
                return slice(n, n + len(item))
            n += len(item)
        raise IndexError("Item not in list")

    else:
        for entry, item in lookup:
            try:
                idx = entry.index(x)
                if idx >= 0:
                    return n + idx
            except AttributeError:
                pass
            n += len(item)

    raise IndexError("Cound not find item")


def flatten_model(model: System, symbols, t):
    # form a list of (for n nodes)
    #   X = [x_{0; 0}, ..., x_{n; k_x}]   (state)
    #   U = [u_{0; 0}, ..., u_{n; k_u}]   (inputs)
    #   Y = [y_{0; 0}, ..., y_{n; k_y}]   (output)
    #   P = [p_{0; 0}, ..., p_{n;_k_p}]   (parameters)
    #   S = [s_{0, 0}, ..., s_{n, k_s}]   (slack variables)
    #
    # split expressions into
    #    \dot{J} = j(X, U, P)
    #    \dot{X} = F(X, U, Z, P)
    #          Y = G_{eq}(X, U, P)
    #          S = G_{ineq}(X, U, P)
    #          0 = G_{dae}(X, U, Z, P)
    #          U = LY
    #

    inverse_map = [(cd.t, t)]

    G_eq = []
    F_eq = []
    Dx_eq = []
    coupling_vars = []
    offset = Signature()
    for component in model.components:
        this_map = []
        this_sig = component.signature
        variables = (
            component.inputs, component.state,
            component.outputs, component.parameters
        )
        for var, symbol, start, length in zip(variables, symbols,
                                              offset, this_sig):
            if not length:
                continue
            this_map.append((var, symbol[start:start + length]))

        for expr in component.expressions():
            assert not is_inequality(expr), \
                "Inequalities within models are not yet supported"

            ode_pairs, g, slack_vars = partition_derivatives(expr)
            for dx, f in ode_pairs:
                Dx_eq.append(dx)
                F_eq.append(f)
            G_eq += g
            coupling_vars += slack_vars

        inverse_map += this_map
        offset += this_sig

    G_eq += [i == j for i, j in model.wires]


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