"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name


if backend_name == 'casadi':
    import sysopt.backends.casadi as backend
    from sysopt.backends.casadi.math import *
else:
    raise NotImplementedError(f'Backend {backend_name} is not implemented')


lambdify = backend.lambdify
InterpolatedPath = backend.InterpolatedPath
Integrator = backend.Integrator
