"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name
from sysopt.backends.impl_hooks import get_implementation


def get_backend():
    if backend_name == 'casadi':
        import sysopt.backends.casadi as backend
    elif backend_name == 'sympy':
        import sysopt.backends.sympy as backend
    else:
        raise NotImplementedError(f'Backend {backend_name} is not implemented')

    return backend


# as_array = backend.as_array
# get_integrator = backend.integrator.get_integrator
# get_variational_solver = backend.get_variational_solver
