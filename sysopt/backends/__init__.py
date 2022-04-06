"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name


if backend_name == 'casadi':
    import sysopt.backends.casadi as backend
    from sysopt.backends.casadi.math import *
else:
    raise NotImplementedError(f'Backend {backend_name} is not implemented')

SymbolicVector = backend.SymbolicVector
lambdify = backend.lambdify
InterpolatedPath = backend.InterpolatedPath
Integrator = backend.Integrator
sparse_matrix = backend.sparse_matrix

list_symbols = backend.list_symbols
concatenate_symbols = backend.concatenate
cast = backend.cast
constant = backend.constant
is_symbolic = backend.is_symbolic
