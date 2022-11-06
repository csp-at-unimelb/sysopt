from sysopt.backends.implementation_hooks import BackendContext


class CasadiBackend(BackendContext):
    name = 'casadi'

    @staticmethod
    def get_integrator(*args, **kwargs):
        from sysopt.backends.casadi import get_integrator
        return get_integrator(*args, **kwargs)

    @staticmethod
    def as_array(*args, **kwargs):
        from sysopt.backends.casadi import as_array
        return as_array(*args, **kwargs)

    @staticmethod
    def get_variational_solver(*args, **kwargs):
        from sysopt.backends.casadi import get_variational_solver
        return get_variational_solver(*args, **kwargs)


class SympyBackend(BackendContext):
    name = 'sympy'

    @staticmethod
    def get_integrator(*args, **kwargs):
        from sysopt.backends.sympy import get_integrator
        return get_integrator(*args, **kwargs)

    @staticmethod
    def as_array(*args, **kwargs):
        from sysopt.backends.sympy import as_array
        return as_array(*args, **kwargs)
