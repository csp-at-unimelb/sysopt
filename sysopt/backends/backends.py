"""Backend bindings."""
from sysopt.backends.implementation_hooks import BackendContext

# pylint: disable=import-outside-toplevel


class CasadiBackend(BackendContext):
    """Casadi Backend"""
    name = 'casadi'
    package = 'sysopt.backends'


class SympyBackend(BackendContext):
    """Sympy Backend"""
    name = 'sympy'
    package = 'sysopt.backends'
