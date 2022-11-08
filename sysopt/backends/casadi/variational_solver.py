"""Optimisation Routines for Variational Problems"""

import casadi
import numpy as np

from collections import namedtuple
from operator import mul
from functools import reduce

from sysopt.backends.casadi.expression_graph import substitute
from sysopt.problems.problem_data import MinimumPathProblem, CollocationSolverOptions
from sysopt.backends.casadi.path import InterpolatedPath


Problem = namedtuple(
    'Problem',
    ['state', 'control', 'parameters', 'running_cost', 'terminal_cost',
     'vector_field', 'initial_state', 'constraints']
)


default_options = CollocationSolverOptions()


def get_collocation_matrices(degree):
    collocation_times = np.append(
        0, casadi.collocation_points(degree, 'legendre')
    )
    collocation_coeeff = np.zeros((degree + 1, degree + 1))
    continuity_coeff = np.zeros(degree + 1)
    quad_coeff = np.zeros_like(continuity_coeff)

    for i in range(degree + 1):
        tau_i = collocation_times[i]
        factors = [
            np.poly1d([1, -tau_j]) / (tau_i - tau_j)
            for tau_j in collocation_times if tau_i != tau_j
        ]
        basis_i = reduce(mul, factors)
        dbasis_i = np.polyder(basis_i)
        continuity_coeff[i] = basis_i(1)
        collocation_coeeff[i, :] = [
            dbasis_i(tau_j) for tau_j in collocation_times
        ]
        quad_coeff[i] = np.polyint(basis_i)(1.0)

    return collocation_times, collocation_coeeff, continuity_coeff, quad_coeff


def _get_solver(t_final: float,
                problem: Problem,
                options: CollocationSolverOptions):

    # Direct Collocation method
    times, colloc_coeff, diff_coeff, quad_coeff = get_collocation_matrices(
        options.polynomial_degree
    )

    params = [casadi.MX.sym(str(p), p.shape) for p in problem.parameters]
    f = casadi.Function(
        'f',
        [problem.state[0], problem.control[0]] + problem.parameters,
        [problem.vector_field, problem.running_cost]
    )

    g = casadi.Function(
        'g',
        [problem.state[0], problem.control[0]] + problem.parameters,
        [casadi.vertcat(*problem.constraints)]
    )

    phi = casadi.Function(
        'phi',
        [problem.state[0], problem.control[0]] + problem.parameters,
        [problem.terminal_cost]
    )

    g_lower = [0]*len(problem.constraints)
    g_upper = [np.inf]*len(problem.constraints)
    steps = int(options.grid_size - 1)
    dt = 1 / steps


    decision_vars = []      # tuple (lb < x < ub)
    constraints = []    # tuple (lb < g(x,u) < ub)

    cost = 0

    dim_x, _ = problem.state[0].shape
    dim_u, _ = problem.control[0].shape

    x_lower, x_upper = problem.state[1]
    u_lower, u_upper = problem.control[1]

    du_upper = [np.inf] * dim_u
    du_lower = [-np.inf] * dim_u

    u_0 = [0] * dim_u

    x = casadi.MX.sym('X_0', dim_x)
    u = casadi.MX.sym('U_0', dim_u)
    x_out = [casadi.vertcat(x, u)]
    x0 = casadi.Function('x0', problem.parameters, [problem.initial_state])
    x_0 = x0(*params)
    decision_vars.append((x_0, x, x_0))
    decision_vars.append((u_lower, u, u_upper))
    decision_vars_guess = [x_0, u_0]

    constraints.append((g_lower, g(x, u, *params), g_upper))

    for k in range(steps):
        collocation_points = []
        du = casadi.MX.sym(f'dU_{k}', dim_u)
        decision_vars_guess.append([0] * dim_u)
        decision_vars.append((du_lower, du, du_upper))
        for j in range(options.polynomial_degree):
            x_jk = casadi.MX.sym(f'X_{j},{k}', dim_x)
            decision_vars.append((x_lower, x_jk, x_upper))
            decision_vars_guess.append(x_0)
            collocation_points.append(x_jk)

        x_next = diff_coeff[0] * x
        for j in range(1, options.polynomial_degree + 1):
            # Todo: Check if this indexing is right as C[:, 0] is never used
            dx = colloc_coeff[0, j] * x
            dx += sum(
                c_ij * x_ij
                for c_ij, x_ij in zip(colloc_coeff[1:, j], collocation_points)
            )
            u_k = u + du * times[j - 1] * dt
            f_inter, q_inter = f(collocation_points[j - 1], u_k, *params)
            constraints.append(([0] * dim_x, dt * f_inter - dx, [0] * dim_x))

            x_next = x_next + diff_coeff[j] * collocation_points[j - 1]
            cost += quad_coeff[j] * q_inter * dt

        u_next = u + dt * du

        x = casadi.MX.sym(f'X_{k + 1}', dim_x)
        u = casadi.MX.sym(f'U_{k + 1}', dim_u)
        decision_vars.append((x_lower, x, x_upper))
        decision_vars.append((u_lower, u, u_upper))
        decision_vars_guess.append(x_0)
        decision_vars_guess.append(u_0)
        constraints.append(
            ([0] * dim_x, x - x_next, [0] * dim_x)
        )
        constraints.append((
            [0]*dim_u, u - u_next, [0] * dim_u
        ))

        x_out.append(casadi.vertcat(x, u))
        constraints.append((g_lower, g(x, u, *params), g_upper))

    cost += phi(x, u, *params)

    x_min, x_array, x_max = zip(*decision_vars)
    c_min, c_array, c_max = zip(*constraints)

    x_min = casadi.vertcat(*x_min)
    x_max = casadi.vertcat(*x_max)
    c_min = casadi.vertcat(*c_min)
    c_max = casadi.vertcat(*c_max)

    x_array = casadi.vertcat(*x_array)
    c_array = casadi.vertcat(*c_array)
    x_path = casadi.horzcat(*x_out)

    nlp_spec = {
        'f': cost,
        'x': x_array,
        'g': c_array,
        'p': casadi.vertcat(*params)
    }
    nlp_options = {}
    npl_solver = casadi.nlpsol('problems', 'ipopt', nlp_spec, nlp_options)
    x_initial = casadi.vertcat(*decision_vars_guess)
    param_to_args = casadi.Function(
        'nlp_args',
        params,
        [x_initial, casadi.vertcat(*params),  x_min, x_max, c_min, c_max,
         casadi.MX.zeros(x_initial.shape), casadi.MX.zeros(c_max.shape)],
        [str(p) for p in params],
        ['x0', 'p', 'lbx', 'ubx', 'lbg', 'ubg', 'lam_x0', 'lam_g']
    )
    t_path = casadi.MX([i * dt for i in range(steps + 1)])

    soln_to_path = casadi.Function(
        'trajectories',
        [x_array], [t_path, x_path],
        ['X'], ['t', 'x']
    )
    return param_to_args, npl_solver, soln_to_path


class VariationalIntegrator:
    """Function wrapper for an optimisation problem sovler."""
    def __init__(self, t_final, problem, options):
        prob = casadify_problem(problem)
        self._param_map, self._solver, self._post_processing = _get_solver(
            t_final, prob, options
        )

    def __call__(self,  parameters):

        solver_args = self._param_map(parameters)
        solution, *_ = self._solver(*solver_args)
        t, x = self._post_processing(solution)

        return InterpolatedPath('x', t.T, x)


def get_variational_solver(problem: MinimumPathProblem,
                               options=default_options):

    # choose initial number of co-location points per control update

    def func(t_final):
        return VariationalIntegrator(t_final, problem, options)
    return func


def casadify_problem(problem: MinimumPathProblem):

    if isinstance(problem.parameters, list):
        parameters = problem.parameters
    else:
        parameters = [problem.parameters]
    all_symbols = set([problem.state[0], problem.control[0]] + parameters)
    substitutions = {
        v: casadi.MX.sym(f'{v.name}', len(v))
        for v in all_symbols
    }
    non_negative_constraints = [
        substitute(c.to_graph(), substitutions)
        for c in problem.constraints
    ] if problem.constraints else []

    try:
        x_bounds = problem.state[1]
    except IndexError:
        x_bounds = (
            [-np.inf] * len(problem.state[0]),
            [np.inf] * len(problem.state[0])
        )

    try:
        u_bounds = problem.control[1]
    except IndexError:
        u_bounds = (
            [-np.inf] * len(problem.control[0]),
            [np.inf] * len(problem.control[0])
        )

    return Problem(
        state=(substitutions[problem.state[0]], x_bounds),
        control=(substitutions[problem.control[0]], u_bounds),
        parameters=[substitutions[p] for p in parameters],
        initial_state=substitute(problem.initial_state, substitutions),
        vector_field=substitute(problem.vector_field, substitutions),
        constraints=non_negative_constraints,
        running_cost=substitute(problem.running_cost, substitutions),
        terminal_cost=substitute(problem.terminal_cost, substitutions),
    )
