"""Casadi Backend Implementation."""
import casadi as _casadi
import numpy as np
from sysopt.backends.casadi.math import fmin, fmax, heaviside

epsilon = 1e-9


class InterpolatedPath(_casadi.Callback):
    def __init__(self, name, t, x, opts={}):
        super().__init__()
        self.t = t
        self.x = x
        self.construct(name, opts)

    def __len__(self):
        return self.t.shape[1]

    def __getitem__(self, item):
        return self.t[item], self.x[:, item]

    @property
    def shape(self):
        return self.x.shape

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_out(self, i):
        return _casadi.Sparsity.dense((self.x.shape[0], 1))

    def eval(self, arg):
        t = fmax(fmin(arg[0], self.t[-1]), self.t[0])

        dt = self.t[1:] - self.t[:-1]
        dh = 1 / dt
        dx = self.x[:, 1:] - self.x[:, :-1]

        DX = dx * _casadi.repmat(dh, (dx.shape[0], 1))
        m = _casadi.horzcat(dx[:, 0] * dh[0],
                    0.5 * (DX[: 1:] + DX[:, :-1]),
                    dx[:, -1] * dh[-1])

        T = self.t - t
        r = -T[:-1] / dt
        s = 1 - r
        window = (heaviside(T[1:]) - heaviside(T[:-1]))

        rrr = window * r * r * r
        rrs = window * r * r * s
        ssr = window * r * s * s
        sss = window * s * s * s

        P_0 = self.x[:, :-1]

        P_1 = P_0 + m[:, :-1] / 3
        P_3 = self.x[:, 1:]
        P_2 = P_3 - m[:, 1:] / 3

        # Bezier Curve / Cubic Hermite Interpolation
        result = P_0 @ sss.T\
                 + 3 * P_1 @ ssr.T \
                 + 3 * P_2 @ rrs.T\
                 + P_3 @ rrr.T\
                 + heaviside(self.t[0] - t) * self.x[:, 0]\
                 + heaviside(t - self.t[-1]) * self.x[:, -1]

        return [result]


class Integrator:
    """A casadi based solve for the given system."""

    def __init__(self, t, system, resolution=50):
        assert system is not None
        assert system.X is not None
        assert system.f is not None
        # pylint: disable=invalid-name
        self._T = _casadi.SX.sym('T', 1, 1)

        dae_spec = {
            'x': _casadi.vertcat(t, system.X),
            'p': _casadi.vertcat(self._T, system.P),
            'ode': self._T * _casadi.vertcat(_casadi.SX.ones(1, 1), system.f),
        }
        self.x0 = _casadi.Function(
            'x0',
            [system.P],
            [_casadi.vertcat(_casadi.SX.zeros(1, 1), system.X0)]
        )
        self.g = _casadi.Function(
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
        self.f = _casadi.integrator('F', 'idas', dae_spec, solver_options)

    def __call__(self, t, p):
        """Integrate from 0 to t"""

        x0 = self.x0(p)
        z0 = [0] * self.n_alg
        p_prime = _casadi.vertcat(t, p)
        soln = self.f(x0=x0, p=p_prime, z0=z0)

        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p)

        return InterpolatedPath('y', tf, y)


def lambdify(expressions, arguments, name='f'):
    try:
        outputs = [concatenate(expr) for expr in expressions]
    except Exception:
        outputs = [expressions]
    return _casadi.Function(name, arguments, outputs)


def sparse_matrix(shape):
    return _casadi.SX(*shape)


class SymbolicVector(_casadi.SX):
    """Wrapper around SX for vectors."""
    __names = {}

    def __init__(self, *args, **kwarg):
        super().__init__()

    def __repr__(self):
        return self.__name

    def __new__(cls, name, length=1):
        assert isinstance(length, int)
        obj = SymbolicVector.sym(name, length)
        try:
            idx = SymbolicVector.__names[name]
        except KeyError:
            SymbolicVector.__names[name] = 0
            idx = 0

        SymbolicVector.__names[name] += 1
        obj.__name = f'{name}_{idx}'
        obj.__class__ = cls
        if cls is not SymbolicVector:
            obj.__bases__ = [SymbolicVector, _casadi.SX]
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
        return 1


def list_symbols(expr) -> set:
    return set(_casadi.symvar(expr))


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
    if not isinstance(v0, _casadi.SX):
        result = cast(v0)
    else:
        result = v0
    for v_i in v_n:
        if v_i is not None:
            result = _casadi.vertcat(result, v_i)

    return result


def cast(arg):
    if arg is None:
        return None
    if isinstance(arg, (float, int)):
        return _casadi.SX(arg)
    if isinstance(arg, (_casadi.SX, _casadi.MX, _casadi.DM)):
        return arg
    elif isinstance(arg, (list, tuple, np.ndarray)):
        return _casadi.SX(arg)

    raise NotImplementedError(f'Don\'t know how to cast {arg.__class__}')


def is_symbolic(arg):
    if hasattr(arg, 'is_symbolic'):
        return arg.is_symbolic
    return isinstance(arg, _casadi.SX)


def constant(value):
    assert isinstance(value, (int, float))
    c = _casadi.SX(value)
    return c

