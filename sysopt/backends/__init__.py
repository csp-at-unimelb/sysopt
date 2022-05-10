"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name


if backend_name == 'casadi':
    import sysopt.backends.casadi as backend
    from sysopt.backends.casadi.math import *
    Integrator = backend.Integrator
    sparse_matrix = backend.sparse_matrix
    is_symbolic = backend.is_symbolic
elif backend_name == 'sympy':
    import sysopt.backends.sympy as backend
    from sysopt.backends.sympy.math import *
    KO_inputs = backend.KO_inputs
else:
    raise NotImplementedError(f'Backend {backend_name} is not implemented')

SymbolicVector = backend.SymbolicVector
InterpolatedPath = backend.InterpolatedPath
list_symbols = backend.list_symbols
lambdify = backend.lambdify

concatenate_symbols = backend.concatenate
cast = backend.cast
constant = backend.constant
