"""Module for differentiable path solvers."""

from dataclasses import asdict

import casadi as _casadi
import numpy as np

from sysopt.backends.casadi.math import heaviside
from sysopt.backends.casadi.expression_graph import lambdify
from sysopt.types import Domain


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
        self.t = _casadi.DM(t) if isinstance(t, np.ndarray) else t
        self.x = _casadi.DM(x) if isinstance(x, np.ndarray) else x

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
        # return self.linearly_interpolate(arg[0])
        return self.cubic_interpolate(arg[0])

    def linearly_interpolate(self, t):

        t_lower = t - self.t[:-1]
        t_upper = t - self.t[1:]
        dt = t_upper - t_lower
        window = heaviside(t_lower) - heaviside(t_upper)
        start = heaviside(self.t[0] - t)
        end = heaviside(t - self.t[-1])

        s_lower = window * t_upper / dt
        s_upper = - window * t_lower / dt

        x_start = start * self.x[:, 0]
        x_end = end * self.x[:, -1]
        x_t = (
            self.x[:, :-1] @ s_lower.T + self.x[:, 1:] @ s_upper.T
            + x_start + x_end
        )

        return [x_t]

    def cubic_interpolate(self, t):
        # pylint: disable=invalid-name

        n = self.x.shape[0]

        t_lower = t - self.t[:-1]
        t_upper = t - self.t[1:]

        dt = t_upper - t_lower
        window = heaviside(t_lower) - heaviside(t_upper)
        start = heaviside(self.t[0] - t)
        end = heaviside(t - self.t[-1])

        s_lower = t_upper / dt
        s_upper = - t_lower / dt

        x_start = start * self.x[:, 0]
        x_end = end * self.x[:, -1]

        T = _casadi.repmat(self.t, (n, 1))

        dx = _casadi.horzcat(
            (self.x[:, 1] - self.x[:, 0]) / (T[:, 1] - T[:, 0]),
            (self.x[:, 2:] - self.x[:, :-2]) / (T[:, 2:] - T[:, :-2]) / 2,
            (self.x[:, -1] - self.x[:, -2]) / (T[:, -1] - T[:, -2])
        )

        c_0 = window * s_lower * s_lower * s_lower
        c_1 = window * s_lower * s_lower * s_upper
        c_2 = window * s_lower * s_upper * s_upper
        c_3 = window * s_upper * s_upper * s_upper

        p0 = self.x[:, :-1]
        p1 = self.x[:, :-1] + dx[:, :-1] / 3
        p2 = self.x[:, 1:] - dx[:, 1:] / 3
        p3 = self.x[:, 1:]

        x_t = (
            p0 @ c_0.T + 3 * p1 @ c_1.T + 3 * p2 @ c_2.T + p3 @ c_3.T
            + x_start + x_end
        )

        return [x_t]


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


def generate_dae_from(flattened_system, quadratures):
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
    p_name = list(symbols.keys())[-1]
    x0 = evaluate(flattened_system.initial_conditions, {p_name: p})
    x0_impl = _casadi.vertcat(_casadi.SX.zeros(1, 1), x0)
    initial_conditions = _casadi.Function('x0', [p], [x0_impl])

    if flattened_system.constraints is not None:
        assert z.size() != (0, 1)
        h_impl = evaluate(flattened_system.constraints, symbols)
        h = _casadi.Function('h', [t, x, z, p], [h_impl])
        h0_impl = h(0, x0, z, p)
        objective = _casadi.Function('norm_z0', [z, p], [h0_impl])
        z0_func = _casadi.rootfinder('z0', 'newton', objective)
        spec.update(dict(alg=h_impl, z=z))
        solver = 'idas'
    else:
        assert z.size() == (0, 1)
        solver = 'idas'
        z0_func = None

    if quadratures:
        integrand = quadratures.vector_quadrature.call(
            {quadratures.output_variable: flattened_system.output_map}
        )
        q_dot = t_final * evaluate(integrand, symbols)
        spec.update(dict(quad=q_dot))

    output_map = _casadi.Function(
        'g', [t, x, z, p], [evaluate(flattened_system.output_map, symbols)]
    )

    return solver, spec, initial_conditions, z0_func, output_map, symbols


class Integrator:
    """A casadi based solve for the given system."""
    def __init__(self, system, resolution=50, quadratures=None):
        solver_options = {
            'grid': [i / resolution for i in range(resolution + 1)],
            'output_t0': True
        }
        data = generate_dae_from(system, quadratures)
        solver, self.dae_spec, self.x0, self.z0, self.g, _ = data

        self.f = _casadi.integrator('F', solver, self.dae_spec, solver_options)
        self.domain = system.domain
        self.quadratures = quadratures

    def pushforward(self, t, p, dp):
        p_sym = _casadi.MX.sym('p', len(p))
        t_sym = _casadi.MX.sym('t')
        p_prime = _casadi.vertcat(t_sym, p_sym)
        x0 = self.x0(p_sym)
        z0_est = [0] * self.domain.constraints

        z0 = self.z0(z0_est, p_sym) if self.z0 else z0_est

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
        z0_est = [0] * self.domain.constraints
        z0 = self.z0(z0_est, p) if self.z0 else z0_est

        p_prime = _casadi.vertcat(t, p)
        soln = self.f(x0=x0, p=p_prime, z0=z0)

        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']
        y = self.g(tf, x, z, p)

        if self.quadratures:
            q = soln['qf']
            return InterpolatedPath('y', tf, y), InterpolatedPath('q', tf, q)
        return InterpolatedPath('y', tf, y)

    def __call__(self, t, p):
        """Integrate from 0 to t"""
        soln = self.integrate(t, p)
        if self.quadratures:
            y, q = soln
            return y(t), q(t)
        else:
            return soln(t)
