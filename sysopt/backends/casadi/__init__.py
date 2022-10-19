"""Casadi Backend Implementation."""
import casadi as _casadi
from sysopt.backends.casadi.expression_graph import substitute
from sysopt.backends.casadi.path import InterpolatedPath
from sysopt.backends.casadi.integrator import get_integrator
from sysopt.backends.casadi.variational_solver import get_variational_solver
import sysopt.backends.casadi.codesign_solver
import sysopt.backends.casadi.foreign_function
from sysopt.backends.casadi.math import *
from sysopt.backends.impl_hooks import get_implementation
epsilon = 1e-9


def sparse_matrix(shape):
    return _casadi.MX(*shape)


def list_symbols(expr) -> set:
    return set(_casadi.symvar(expr))


def as_array(item):
    if isinstance(item, _casadi.DM):
        return item.full()

    raise TypeError(f'Don\'t know how to cast {item} to an array')
