"""Casadi Backend Implementation."""
import casadi as _casadi

from sysopt.backends.casadi.expression_graph import substitute
from sysopt.backends.casadi.path import InterpolatedPath
from sysopt.backends.casadi.integrator import Integrator
from sysopt.backends.casadi.variational_solver import get_variational_integrator
epsilon = 1e-9


def sparse_matrix(shape):
    return _casadi.SX(*shape)


def list_symbols(expr) -> set:
    return set(_casadi.symvar(expr))

