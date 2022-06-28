"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name


if backend_name == 'casadi':
    import sysopt.backends.casadi as backend
    from sysopt.backends.casadi.math import *
else:
    raise NotImplementedError(f'Backend {backend_name} is not implemented')


InterpolatedPath = backend.InterpolatedPath
get_integrator = backend.integrator.get_integrator
