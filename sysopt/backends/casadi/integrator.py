"""Symbolic DAE Integrator and builders"""

import casadi
from sysopt.backends.casadi.path import InterpolatedPath
from sysopt.backends.casadi.expression_graph import substitute


def get_integrator(system, resolution=50, quadratures=None):
    if system.state_transitions is None:
        return Integrator(system, resolution, quadratures)
    else:
        raise NotImplementedError


class Integrator:
    """A casadi based solve for the given system."""
    def __init__(self, system, resolution=50, quadratures=None):
        solver_options = {
            'grid': [i / resolution for i in range(resolution + 1)],
            'output_t0': True
        }
        data = generate_dae_from(system, quadratures)
        solver, self.dae_spec, self.x0, self.z0, self.g, _ = data

        self.f = casadi.integrator('F', solver, self.dae_spec, solver_options)
        self.domain = system.domain
        self.quadratures = quadratures

    def pushforward(self, t, p, dp):
        p_sym = casadi.MX.sym('p', len(p))
        t_sym = casadi.MX.sym('t')
        p_prime = casadi.vertcat(t_sym, p_sym)
        x0 = self.x0(p_sym)
        z0_est = [0] * self.domain.constraints

        z0 = self.z0(z0_est, p_sym) if self.z0 else z0_est

        f = casadi.integrator('F', 'idas', self.dae_spec)
        soln = f(x0=x0, p=p_prime, z0=z0)
        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p_sym)
        dp_sym = casadi.MX.sym('dp', len(dp))
        dy_symbol = casadi.jtimes(y, p_sym, dp_sym)

        if not self.quadratures:
            dy = casadi.Function('dY', [t_sym, p_sym, dp_sym], [y, dy_symbol])
            return dy(t, p, dp)
        else:
            q = soln['qf']
            dq_symbol = casadi.jtimes(q, p_sym, dp_sym)
            dq = casadi.Function('dQ', [t_sym, p_sym, dp_sym],
                                 [y, q, dy_symbol, dq_symbol])
            result = dq(t, p, dp)
            return result

    def integrate(self, t, p):
        x0 = self.x0(p)
        z0_est = [0] * self.domain.constraints
        z0 = self.z0(z0_est, p) if self.z0 else z0_est

        p_prime = casadi.vertcat(t, p)
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


def generate_dae_from(flattened_system, quadratures):
    symbols = {
        s: casadi.SX.sym(f'{s.name}', len(s))
        for s in flattened_system.output_map.arguments
    }

    t, x, z, u, p = symbols.values()
    zu = casadi.vertcat(z, u)
    t_final = casadi.SX.sym('T', 1, 1)

    f_impl = t_final * casadi.vertcat(
        casadi.SX.ones(1, 1),
        substitute(flattened_system.vector_field, symbols)
    )

    spec = dict(
        x=casadi.vertcat(t, x),
        p=casadi.vertcat(t_final, p),
        ode=f_impl
    )

    x0 = substitute(flattened_system.initial_conditions, symbols)

    x0_impl = casadi.vertcat(casadi.SX.zeros(1, 1), x0)
    initial_conditions = casadi.Function('x0', [p], [x0_impl])

    if flattened_system.constraints is not None:
        h_impl = substitute(flattened_system.constraints, symbols)
        h = casadi.Function('h', [t, x, zu, p], [h_impl])
        h0_impl = h(0, x0, zu, p)
        objective = casadi.Function('norm_z0', [zu, p], [h0_impl])
        z0_func = casadi.rootfinder('z0', 'newton', objective)
        spec.update(dict(alg=h_impl, z=zu))
        solver = 'idas'
    else:
        solver = 'idas'
        z0_func = None

    if quadratures:
        integrand = quadratures.vector_quadrature.call(
            {quadratures.output_variable: flattened_system.output_map.graph}
        )

        q_dot = t_final * substitute(integrand, symbols)
        spec.update(dict(quad=q_dot))

    output_map = casadi.Function(
        'g', [t, x, zu, p], [substitute(flattened_system.output_map, symbols)]
    )

    return solver, spec, initial_conditions, z0_func, output_map, symbols
