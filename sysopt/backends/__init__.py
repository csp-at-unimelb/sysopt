"""Symbolic Backend Loader"""

from sysopt.env import backend as backend_name


if backend_name == 'casadi':
    import sysopt.backends.casadi as backend
    from sysopt.backends.casadi.math import *
else:
    raise NotImplementedError(f'Backend {backend_name} is not implemented')


InterpolatedPath = backend.InterpolatedPath
as_array = backend.as_array
get_integrator = backend.integrator.get_integrator
get_variational_solver = backend.get_variational_integrator
to_function = backend.to_function
function_from_graph = to_function
