"""Casadi Backend Implementation."""
import casadi
from casadi import *
import numpy as np

from sysopt.helpers import InterpolatedPath


class Integrator:
    """A casadi based solve for the given system."""

    def __init__(self, t, system, resolution=50):
        assert system is not None
        assert system.X is not None
        assert system.f is not None
        # pylint: disable=invalid-name
        self._T = casadi.SX.sym('T', 1, 1)

        dae_spec = {
            'x': vertcat(t, system.X),
            'p': casadi.vertcat(self._T, system.P),
            'ode': self._T * vertcat(SX.ones(1, 1), system.f),
        }
        self.x0 = Function(
            'x0',
            [system.P],
            [vertcat(SX.zeros(1, 1), system.X0)]
        )
        self.g = Function(
            'g',
            [t, system.X, system.Z, system.P],
            [system.g]
        )
        self.n_alg = 0
        if system.Z is not None:
            dae_spec.update({'z': system.Z, 'alg': system.h})
            self.n_alg, _ = system.Z.shape

        solver_options = {
            'grid': [i / resolution for i in range(resolution + 1)],
            'output_t0': True
        }
        self.f = integrator('F', 'idas', dae_spec, solver_options)

    def __call__(self, t, p):
        """Integrate from 0 to t"""

        x0 = self.x0(p)
        z0 = [0] * self.n_alg
        p_prime = casadi.vertcat(t, p)
        soln = self.f(x0=x0, p=p_prime, z0=z0)

        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p)


        return InterpolatedPath(tf, y)


class ProblemEvaluator:
    def __init__(self,
                 problem,
                 flat_system,
                 t_final,
                 decision_variables=None,
                 constants=None
                 ):

        self.system = flat_system
        print(problem.cost)
        print(decision_variables)
        for c in problem.constraints:
            print(c)




    def __call__(self, **decision_variables):
        return None

def sparse_matrix(shape):
    return SX(*shape)


def symbol(*args):
    return CasadiVector(*args)


class CasadiVector(SX):
    """Wrapper around SX for vectors."""
    __names = {}

    def __init__(self, *args, **kwarg):
        super().__init__()

    def __repr__(self):
        return self.__name

    def __new__(cls, name, length=1):
        assert isinstance(length, int)
        obj = CasadiVector.sym(name, length)
        try:
            idx = CasadiVector.__names[name]
        except KeyError:
            CasadiVector.__names[name] = 0
            idx = 0

        CasadiVector.__names[name] += 1
        obj.__name = f'{name}_{idx}'
        obj.__class__ = cls
        if cls is not CasadiVector:
            obj.__bases__ = [CasadiVector, SX]
        return obj

    def __iter__(self):
        return iter(
            [self[i] for i in range(self.shape[0])]
        )

    def __len__(self):
        return self.shape[0]

    def index(self, value):
        for i, v in enumerate(self):
            if v is value:
                return i
        return  1



def list_symbols(expr) -> set:
    return set(symvar(expr))


def concatenate(*vectors):
    """Concatenate arguments into a casadi symbolic vector."""
    try:
        v0, *v_n = vectors
    except ValueError:
        return None
    while v0 is None:
        try:
            v0, *v_n = v_n
        except ValueError:
            return None
    if not isinstance(v0, SX):
        result = cast(v0)
    else:
        result = v0
    for v_i in v_n:
        if v_i is not None:
            result = vertcat(result, v_i)

    return result


def cast(arg):
    if arg is None:
        return None
    if isinstance(arg, (float, int)):
        return SX(arg)
    if isinstance(arg, (SX, MX, DM)):
        return arg
    elif isinstance(arg, (list, tuple, np.ndarray)):
        return SX(arg)

    raise NotImplementedError(f'Don\'t know how to cast {arg.__class__}')


def is_symbolic(arg):
    if hasattr(arg, 'is_symbolic'):
        return arg.is_symbolic
    return isinstance(arg, casadi.SX)


def scalar_constant(value):
    assert isinstance(value, (int, float))
    c = SX(value)
    return c
