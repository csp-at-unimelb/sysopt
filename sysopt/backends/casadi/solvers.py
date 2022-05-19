from dataclasses import asdict

import casadi as _casadi
from sysopt.backends.casadi.math import fmin, fmax, heaviside
from sysopt.backends.casadi.symbols import *
from sysopt.backends.casadi.expression_graph import lambdify
from sysopt.types import Domain
# from sysopt.solver.symbol_database import FlattenedSystem


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


def construct_symbols_from(domain: Domain):
    symbols = {
        name: _casadi.SX.sym(name, length, 1)
        for name, length in asdict(domain).items()
    }
    return symbols


def evaluate(func, symbols):

    if isinstance(func, list):
        return _casadi.vertcat(*[evaluate(f, symbols) for f in func])
    if isinstance(func, (float, int)):
        return _casadi.SX(func)
    try:
        func_symbols = func.symbols()
    except AttributeError:

        return func

    if not func_symbols:
        return func()

    func_args, datum = zip(*[
        (v, symbols[v.name]) for v in func_symbols
        if v.name in symbols
    ])

    f = lambdify(func, func_args, 'f')
    result = f(*datum)

    return result


def generate_dae_from(flattened_system):
    symbols = construct_symbols_from(flattened_system.domain)
    t, x, z, u, p = symbols.values()
    assert u.size() == (0, 1), u.size()
    t_final = _casadi.SX.sym('T', 1, 1)

    f_impl = t_final * _casadi.vertcat(
        _casadi.SX.ones(1, 1),
        evaluate(flattened_system.vector_field, symbols)
    )

    spec = dict(
        x=_casadi.vertcat(t, x),
        p=_casadi.vertcat(t_final, p),
        ode=f_impl
    )
    if flattened_system.constraints is not None:
        assert z.size() != (0, 1)
        h_impl = evaluate(flattened_system.constraints, symbols)
        spec.update(dict(alg=h_impl, z=z))
        solver = 'idas'
    else:
        assert z.size() == (0, 1)
        solver = 'idas'

    if flattened_system.quadratures:
        raise NotImplementedError

    p_name = list(symbols.keys())[-1]

    x0_impl = _casadi.vertcat(
        _casadi.SX.zeros(1, 1),
        evaluate(flattened_system.initial_conditions, {p_name: p})
    )
    initial_conditions = _casadi.Function('x0', [p], [x0_impl])

    output_map = _casadi.Function(
        'g', [t, x, z, p], [evaluate(flattened_system.output_map, symbols)]
    )
    return solver, spec, initial_conditions, output_map


class Integrator:
    """A casadi based solve for the given system."""
    def __init__(self, system, resolution=50):
        solver_options = {
            'grid': [i / resolution for i in range(resolution + 1)],
            'output_t0': True
        }
        solver, self.dae_spec, self.x0, self.g = generate_dae_from(system)
        self.f = _casadi.integrator('F', 'idas', self.dae_spec, solver_options)
        self.domain = system.domain

    def pushforward(self, t, p, dp):
        p_sym = _casadi.MX.sym('p', len(p))
        t_sym = _casadi.MX.sym('t')
        p_prime = _casadi.vertcat(t_sym, p_sym)
        x0 = self.x0(p_sym)
        z0 = [0] * self.domain.constraints
        f = _casadi.integrator('F', 'idas', self.dae_spec)
        soln = f(x0=x0, p=p_prime, z0=z0)
        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p_sym)
        dp_sym = _casadi.MX.sym('dp', len(dp))
        dy_symbol = _casadi.jtimes(y, p_sym, dp_sym)
        dy = _casadi.Function('dY', [t_sym, p_sym, dp_sym], [dy_symbol])

        return dy(t, p, dp)

    def integrate(self, t, p):
        x0 = self.x0(p)
        z0 = [0] * self.domain.constraints
        p_prime = _casadi.vertcat(t, p)
        soln = self.f(x0=x0, p=p_prime, z0=z0)

        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p)
        return InterpolatedPath('y', tf, y)

    def __call__(self, t, p):
        """Integrate from 0 to t"""
        soln = self.integrate(t, p)

        return soln(t)