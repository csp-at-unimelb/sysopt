"""Symbolic Backend Loader"""
from importlib import import_module, invalidate_caches
from sysopt.backends.implementation_hooks import get_implementation


def get_backend():
    return BackendContext.get_backend()


class BackendContext:
    """Loads and keeps track of symbolic backend."""

    _active_backend = None
    _active_name = None

    def __init__(self, name='casadi', pkg='sysopt.backends'):
        self.__name = name
        self.__pkg = pkg

    def __getattribute__(self, item):
        # There is probably a cleaner, less convoluted way to do this
        try:
            return super().__getattribute__(item)
        except AttributeError:
            pass

        try:
            be = get_backend()
            return getattr(be, item)
        except AttributeError as ex:
            raise NotImplementedError(
                f'Backend {self.__active_name} does not currently '
                f'implement {item}!'
            ) from ex

    def __enter__(self):
        assert not BackendContext._active_backend, \
            'Cannot open a new backend context while existing context ' \
            f'{BackendContext._active_backend} is active'
        module = f'{self.__pkg}.{self.__name}'
        BackendContext._active_backend = import_module(module)
        BackendContext._active_name = module
        invalidate_caches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BackendContext._active_backend = None
        BackendContext._active_name = None

    @staticmethod
    def get_backend():
        assert BackendContext._active_backend is not None, 'No backend loaded.'
        return BackendContext._active_backend

    @staticmethod
    def get_implementation(obj):
        return get_implementation(obj)
