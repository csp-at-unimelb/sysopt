"""Casadi Function Factories."""
import importlib
import numpy as np
# from sysopt.symbolic.core import Algebraic

__backends = {}     # pylint: disable=invalid-name


def get_backend(name='casadi'):
    try:
        return __backends[name]
    except KeyError:
        pass

    for cls in BackendContext.__subclasses__():
        if cls.name == name:
            instance = cls()
            __backends[name] = instance
            return instance
    raise NotImplementedError(f'Unknown backend: {name}')


class BackendContext:
    """Loads and keeps track of symbolic backend."""

    name = None
    package = None

    def __init__(self):
        self._implementations = {}
        self.module = importlib.import_module(
            f'{self.package}.{self.name}',
            f'{self.package}'
        )

    def implements(self, sysopt_cls):

        if not isinstance(sysopt_cls, np.ufunc):
            key = sysopt_cls
        else:
            key = str(sysopt_cls)
        def wrapper(func):
            self._implementations[key] = func
            return func
        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_implementation(self, obj):
        if not isinstance(obj, np.ufunc):
            key = type(obj)
        else:
            key = str(obj)
        try:
            factory = self._implementations[key]
        except KeyError as ex:
            msg = f'Backend doesn\'t know how to turn and object of {key}' \
                  'into a function'
            raise NotImplementedError(msg) from ex
        if isinstance(obj, np.ufunc):
            return factory
        else:
            return factory(obj)

    def get_integrator(self, *args, **kwargs):
        return self.module.get_integrator(*args, **kwargs)

    def as_array(self, *args, **kwargs):
        return self.module.as_array(*args, **kwargs)

    def get_variational_solver(self, *args, **kwargs):
        return self.module.get_variational_solver(*args, **kwargs)
