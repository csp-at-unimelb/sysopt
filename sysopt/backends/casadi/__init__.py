"""Casadi Backend Implementation."""
import casadi as _casadi
from sysopt.backends.casadi.math import fmin, fmax, heaviside
from sysopt.backends.casadi.symbols import *


epsilon = 1e-9


class InterpolatedPath(_casadi.Callback):
    """Function class for 1d cubic interpolation between gridpoints.

    Args:
        name: Function name
        t: 1xN Array of data for the independant variable
        x: MxN array of data of M-dimensional vectors at the nth sample point.
        opt: Casadi options.

    """

    # pylint: disable=dangerous-default-value
    def __init__(self,
                 name: str,
                 t,
                 x,        # As per CasADI docs.
                 opts={}):
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

        delta_x = dx * _casadi.repmat(dh, (dx.shape[0], 1))
        m = _casadi.horzcat(dx[:, 0] * dh[0],
                    0.5 * (delta_x[: 1:] + delta_x[:, :-1]),
                    dx[:, -1] * dh[-1])

        t_rel = self.t - t
        r = -t_rel[:-1] / dt
        s = 1 - r
        window = (heaviside(t_rel[1:]) - heaviside(t_rel[:-1]))

        rrr = window * r * r * r
        rrs = window * r * r * s
        ssr = window * r * s * s
        sss = window * s * s * s

        # pylint: disable=invalid-names
        p_0 = self.x[:, :-1]

        p_1 = p_0 + m[:, :-1] / 3
        p_3 = self.x[:, 1:]
        p_2 = p_3 - m[:, 1:] / 3

        # Bezier Curve / Cubic Hermite Interpolation
        result = p_0 @ sss.T\
                 + 3 * p_1 @ ssr.T \
                 + 3 * p_2 @ rrs.T\
                 + p_3 @ rrr.T\
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
    # CasADI api - throws general exception
    # pylint: disable=broad-except
    try:
        outputs = [concatenate(expr) for expr in expressions]
    except Exception:
        outputs = [expressions]
    return _casadi.Function(name, arguments, outputs)


def sparse_matrix(shape):
    return _casadi.SX(*shape)




def list_symbols(expr) -> set:
    return set(_casadi.symvar(expr))

