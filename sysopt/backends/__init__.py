"""Symbolic Backend Loader"""
from importlib import import_module, invalidate_caches
from sysopt.backends.impl_hooks import get_implementation


def get_backend():
    return BackendContext.get_backend()


class BackendContext:
    __active_backend = None
    __active_name = None

    def __init__(self, name='casadi', pkg='sysopt.backends'):
        self.__name = name
        self.__pkg = pkg

    def __enter__(self):
        assert not BackendContext.__active_backend, \
            "Cannot open a new backend context while existing context " \
            f"{BackendContext.__active_backend} is active"
        module = f'{self.__pkg}.{self.__name}'
        BackendContext.__active_backend = import_module(module)
        BackendContext.__active_name = module
        invalidate_caches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BackendContext.__active_backend = None
        BackendContext.__active_name = None

    @staticmethod
    def get_backend():
        return BackendContext.__active_backend
