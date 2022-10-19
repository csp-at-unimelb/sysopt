"""Symbolic Backend Loader"""

from sysopt.backends.impl_hooks import get_implementation

_backend_name = None

backend = None


def get_backend():

    global _backend_name, backend
    if _backend_name is None:
        raise NotImplementedError(
            "Function requires a symbolic context but none is loaded."
            "Try wrapping the outer most scope in `with BackendContext()`")
    if _backend_name == 'casadi':
        import sysopt.backends.casadi as be
        backend = be
    else:
        raise NotImplementedError(
            f'Backend {_backend_name} is not implemented'
        )
    return backend


class BackendContext:
    def __init__(self, name='casadi'):
        self.__name = name

    def __enter__(self):
        global _backend_name
        assert _backend_name is None, "Cannot open a new backend context " \
                                       f"while existing context {_backend_name} " \
                                       f"is active"

        _backend_name = self.__name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _backend_name
        _backend_name = None

    def get_backend(self):
        return get_backend()
