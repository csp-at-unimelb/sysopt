import casadi
import math
from typing import Union, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, namedtuple
import numpy as np

from sysopt.types import Domain
from sysopt.backends.casadi.compiler import implements
from sysopt.backends.casadi.expression_graph import substitute, to_function
from sysopt.symbolic.problem_data import ConstrainedFunctional
from sysopt.backends.casadi.foreign_function import CasadiForeignFunction
from sysopt.backends.casadi.variational_solver import get_collocation_matrices
from sysopt.symbolic.symbols import ConstantFunction, Parameter, \
    Variable, is_temporal, Inequality, PathInequality, GraphWrapper, \
    Function, Compose
# from sysopt.solver.solver import Problem
from sysopt.symbolic.decision_variables import PiecewiseConstantSignal


# issues
# if u(t) is piecewise constant
# then the free-end time problem means that the
# u grid is dependent on state.
__all__ = []


class ConstantFactory:
    def __init__(self,
                 param_or_var: Union[Parameter,Variable],
                 lower_bound, upper_bound
                 ):
        self.symbol = casadi.MX.sym(param_or_var.name,
                                    *param_or_var.shape)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, t):
        return self.symbol

    def finalise(self):
        return self.symbol, \
               self.lower_bound,\
               self.upper_bound, \
               self.initial_guess()

    def initial_guess(self):
        guess = max(self.lower_bound, min(0, self.upper_bound))
        return guess

    def output(self):
        return self.symbol


class PiecewiseConstantFactory:
    def __init__(self, signal: PiecewiseConstantSignal,
                 lower_bound,
                 upper_bound):
        self._shape = signal.shape
        self._name = signal.name
        self._frequency = signal.frequency
        # todo: check if they're the same shape
        self._vector = []
        self.lower = lower_bound
        self.upper = upper_bound

    def __call__(self, t):
        index = max(math.ceil(t * self._frequency) - 1, 0)

        last = len(self._vector)
        while index >= last:
            new_symbol = casadi.MX.sym(f'{self._name}[{last}]', *self._shape)
            self._vector.append(new_symbol)
            last = len(self._vector)

        return self._vector[index]

    def regularisation_cost(self):
        factor = 1 / len(self._vector)
        return factor ** 2 * sum((v2 - v1)**2
                    for v2, v1 in zip(self._vector[1:], self._vector[:-1]))

    def finalise(self):
        x = casadi.vertcat(*self._vector)
        x_lower = np.repeat(self.lower, len(self._vector))
        x_upper = np.repeat(self.upper, len(self._vector))
        x_initial = np.repeat(self.initial_guess(), len(self._vector))

        return x, x_lower, x_upper, x_initial

    def initial_guess(self):
        if self.lower != -np.inf and self.upper != np.inf:
            return (self.upper + self.lower) / 2
        else:
            return max(min(0, self.upper), self.lower)

    def output(self):
        return casadi.vertcat(*self._vector)


_factories = {
    PiecewiseConstantSignal: PiecewiseConstantFactory,
    Variable: ConstantFactory,
    Parameter: ConstantFactory
}


@dataclass
class CodesignSolverOptions:
    final_time: float = 1
    degree: int = 4
    grid_size: int = 100
    solver: str = "ipopt"
    solver_options: Optional[Dict] = None


@dataclass
class CodesignSolution:
    cost: float
    argmin: Dict[Union[Parameter, Variable, PiecewiseConstantSignal],
                 Union[float, np.ndarray]]
    t: np.ndarray
    y: np.ndarray
    q: np.ndarray
# info: dict


@implements(ConstrainedFunctional)
def build_codesign_problem(problem: ConstrainedFunctional):
    # casadi functions
    # - B = [dot{x} - f(x,z,p), h(x,z,p), dot{q} - l(x,z,p)]
    if (not isinstance(problem.final_time, ConstantFunction) and
            any(isinstance(p, PiecewiseConstantSignal)
                for p in problem.parameters)):
        raise NotImplementedError(
            'Variational codesign problems with free end time are not currently'
            'supported')

    return FixedTimeCodesignProblem(problem)


# 1 function that evaluates the codesign problem
# this is set up as follows
# minimise |p - p^*|
# such that
#    F(t, x, xdot, z, u, p) = 0
#    dot(q) - h(q) = 0

class ParameterFactory:
    def __init__(self, parameters):
        self.factories = [
            _factories[type(p)](p, *bounds)
            for p, bounds in parameters.items()
        ]

    def __call__(self, t):
        return casadi.vertcat(*[f(t) for f in self.factories])

    def regularisation_cost(self):
        cost = 0
        for factory in self.factories:
            try:
               cost += factory.regularisation_cost()
            except AttributeError:
                pass
        return cost

    def finalise(self):
        p = []
        p_min = []
        p_max = []
        p_0 = []
        for f in self.factories:
            pf, pf_min, pf_max, pf0 = f.finalise()
            p.append(pf)
            p_min.append(pf_min)
            p_max.append(pf_max)
            p_0.append(pf0)

        result = (
            casadi.vertcat(*p),
            casadi.vertcat(*p_min),
            casadi.vertcat(*p_max),
            casadi.vertcat(*p_0)
        )
        for r in result[1:]:
            assert r.is_constant(), r
        return result

    def output_list(self):
        return [factory.output() for factory in self.factories]


class StateFactory:
    def __init__(self, problem_data: 'CasadiCodesignProblemData',
                 t_grid: np.ndarray, p_guess):
        domain = problem_data.domain

        self.dim_x = domain.states
        self.dim_zu = domain.constraints + domain.inputs
        self._xz_0 = self._get_initial_conditions(
            problem_data, t_grid, p_guess)

        self._xz = [
            (casadi.MX.sym(f'x_{i}', self.dim_x),
             casadi.MX.sym(f'z_{i}', self.dim_zu))
            for i in range(len(t_grid))
        ]

        self._xz_colloc = {}

#        self.integrator =

    @staticmethod
    def _get_initial_conditions(problem_data, t_grid, p_guess):
        x0 = problem_data.initial_conditions(p_guess)
        dim_x = problem_data.domain.states

        dim_z = problem_data.domain.constraints + problem_data.domain.inputs
        z = casadi.MX.sym('z', dim_z)
        residue = problem_data.algebraic_constraint(0, x0, z, p_guess)
        objective = casadi.Function('z0', [z], [residue])

        z0 = casadi.rootfinder('z0', 'newton', objective)([0] * dim_z)
        t = casadi.MX.sym('t')
        x = casadi.MX.sym('x', dim_x)
        p = casadi.MX.sym('p', len(p_guess))
        t_final = t_grid[-1]

        f_impl = t_final * casadi.vertcat(
            casadi.MX.ones(1, 1),
            problem_data.vector_field(t, x, z, p)
        )
        h_impl = problem_data.algebraic_constraint(t, x, z, p)

        dae_spec = {
            'x': casadi.vertcat(t, x),
            'z': z,
            'p': p,
            'ode': f_impl,
            'alg': h_impl
        }
        dae_options = {
            'grid': (t_grid / t_final).tolist(),
            'output_t0': True
        }
        integrator = casadi.integrator('dae', 'idas', dae_spec, dae_options)
        soln = integrator(x0=casadi.vertcat(0, x0), z0=z0, p=p_guess)
        xf = soln['xf'][1:, :]
        zf = soln['zf']

        xz = casadi.vertcat(xf, zf)
        return casadi.horzsplit(xz)

    def new_collocation_points(self, step, degree):
        xz = [(casadi.MX.sym(f'x_{step};{d}', self.dim_x),
               casadi.MX.sym(f'z_{step};{d}', self.dim_zu))
              for d in range(degree)]
        x, z = zip(*xz)
        self._xz_colloc[step] = xz
        return x, z

    def new_terminal(self, step):
        return self._xz[step]

    def finalise(self):
        vars = []
        x0 = []
        for i, (x, z) in enumerate(self._xz):
            vars += [x, z]
            x0.append(self._xz_0[i])
            if i + 1 == len(self._xz):
                break
            for xc, zc in self._xz_colloc[i]:
                vars += [xc, zc]
                x0.append(self._xz_0[i])

        x = casadi.vertcat(*vars)
        x0 = np.concatenate(x0)
        x_min = -np.inf * np.ones_like(x0)
        x_max = np.inf * np.ones_like(x0)
        return x, x_min, x_max, x0


def make_parameter_map(problem):
    """Returns a mapping from problem parameters to system parameters"""
    symbols = {
         p: casadi.MX.sym(f'{p.name}', len(p)) for p in problem.parameters
    }
    p_in = casadi.vertcat(*symbols.values())
    p_arg, = problem.parameter_map.symbols()

    system_p = substitute(problem.parameter_map, {p_arg: p_in})

    p_consts = casadi.vertcat(*[
        v for p, v in symbols.items()
        if not isinstance(p, PiecewiseConstantSignal)]
    )

    p_signals = casadi.vertcat(*[
        v for p, v in symbols.items()
        if isinstance(p, PiecewiseConstantSignal)]
    )

    p_system = casadi.Function('parameter_map',
                               [p_consts, p_signals], [system_p])
    p_problem = casadi.Function('problem_map',
                                [p_consts, p_signals],
                                [p_in])
    return p_system, p_problem


def get_initial_conditions(problem, initial_parameters):
    try:
        p, = problem.system.initial_conditions.symbols()
    except ValueError:
        return problem.system.initial_conditions()

    return substitute(problem.system.initial_conditions,
                      {p: initial_parameters})


def transform_constraints(problem: ConstrainedFunctional):
    box_constraints: Dict[Variable, List] = {
        p: [-np.inf,  np.inf] for p in problem.parameters
    }
    point_constraints: List = []
    path_constraints: List = []

    for constraint in problem.constraints:
        if isinstance(constraint, PathInequality):
            raise NotImplementedError
        elif isinstance(constraint, Inequality):
            if constraint.smaller in problem.parameters and \
                    isinstance(constraint.bigger, (int, float, np.ndarray)):
                box_constraints[constraint.smaller][0] = max(
                    box_constraints[constraint.smaller][0],
                    constraint.bigger
                )
            elif constraint.bigger in problem.parameters and \
                    isinstance(constraint.bigger, (int, float, np.ndarray)):
                box_constraints[constraint.smaller][1] = min(
                    box_constraints[constraint.smaller][1],
                    constraint.bigger
                )
            else:
                point_constraints.append(constraint.to_graph())
        else:
            path_constraints.append(constraint.to_graph())

    return box_constraints, point_constraints, path_constraints


@dataclass
class CasadiCodesignProblemData:
    domain: Domain

    vector_field: casadi.Function
    """Function of t, x, zu, p -> dx"""

    outputs: casadi.Function
    """Function of t, x, zu, p-> y"""

    algebraic_constraint: casadi.Function
    """Function of t, x, zu, p-> 0"""

    quadrature: casadi.Function
    """Function of t, y, p -> dot(q)"""

    initial_conditions: casadi.Function
    """Function p -> x(0)"""

    cost_function: casadi.Function
    """Function of T, y[T], q[T] p"""

    parameters: Dict[Union[Variable,Parameter, PiecewiseConstantSignal],
                     Tuple[float, float]]
    """List of all parameters with the upper and lower bounds"""

    path_constraints: casadi.Function
    """Function c(t,y,q) such that c >= 0 implies constraint is satisfied"""

    terminal_constraints: casadi.Function
    """Function C(T y(T), q(T) such"""

    #
    # solution_map: casadi.Function
    # """Map from p_signal and p_const to the actual output data"""


def build_fixed_endpoint_codesign_problem(problem: ConstrainedFunctional,
                                          options=CodesignSolverOptions()
                                          ) -> CasadiCodesignProblemData:

    p = casadi.vertcat(*[
        casadi.MX.sym(str(param), *param.shape)
        for param in problem.parameters
    ])
    flattened_system = problem.system

    symbols = {
        s: casadi.MX.sym(f'{s.name}', len(s))
        for s in flattened_system.output_map.arguments[:-1]
    }

    p_inner = flattened_system.output_map.arguments[-1]
    p_arg, = problem.parameter_map.symbols()
    symbols[p_inner] = substitute(problem.parameter_map.graph, {p_arg: p})
    dx = substitute(flattened_system.vector_field.graph, symbols)
    y = substitute(flattened_system.output_map.graph, symbols)
    try:
        x0 = substitute(flattened_system.initial_conditions.graph, symbols)
    except AttributeError:
        x0 = flattened_system.initial_conditions()

    t, x, z, u, _ = symbols.values()
    zu = casadi.vertcat(z, u)
    args = [t, x, zu, p]

    f = casadi.Function('f', args, [dx])
    g = casadi.Function('g', args, [y])
    x0 = casadi.Function('x0', [p], [x0])

    if flattened_system.constraints:
        res = substitute(flattened_system.constraints.graph, symbols)
    else:
        res = casadi.MX()
    h = casadi.Function('h', args, [res])
    # build quadratures

    symbols = {
        s: casadi.MX.sym(f'{str(s)}', *s.shape)
        for s in problem.value.arguments
    }

    cost_args = list(symbols.values())

    cost_impl = substitute(problem.value.graph, symbols)
    cost = casadi.Function('cost', cost_args, [cost_impl])

    t, y, q, p = cost_args
    q_args = [t, y, p]
    q_dot = casadi.Function(
        'q_dot',
        q_args,
        [substitute(problem.quadratures.graph, symbols)
         if problem.quadratures else casadi.MX()],
    )

    path_constraints = [
        substitute(c.graph, symbols) for c in problem.path_constraints
    ]
    c_t = casadi.Function(
        'c_t', cost_args,
        [casadi.vertcat(*path_constraints) if path_constraints else casadi.MX()]
    )
    point_constraints = [
        substitute(c.graph, symbols) for c in problem.point_constraints
    ]
    c_T = casadi.Function(
        'c_T', cost_args,
        [casadi.vertcat(*point_constraints) if point_constraints else casadi.MX()]
    )

    data = CasadiCodesignProblemData(
        domain=problem.system.domain,
        vector_field=f,
        outputs=g,
        algebraic_constraint=h,
        initial_conditions=x0,
        cost_function=cost,
        parameters=problem.parameters,
        quadrature=q_dot,
        path_constraints=c_t,
        terminal_constraints=c_T,
    )
    t_f = float(problem.final_time())
    grid_size = 100
    for parameter in problem.parameters:
        try:
            grid_size = max(np.ceil(t_f* parameter.frequency), grid_size)
        except AttributeError:
            pass

    hessian_functions = [
        *problem.path_constraints, problem.system.vector_field,
        problem.system.output_map, problem.system.constraints
    ]
    options = CodesignSolverOptions(
        degree=4,
        final_time=float(problem.final_time()),
        grid_size=int(grid_size),
        solver='ipopt',
        solver_options={}
    )
    for f in [f_i for f_i in hessian_functions if f_i is not None] :
        if any(isinstance(node, (Function, Compose))
               for node in f.graph.nodes):
            options.solver_options.update(
                {'ipopt.hessian_approximation': 'limited-memory'}
            )
            print("Using Hessian Approximation")
            break
    return data, options


data = namedtuple('data', ['symbol', 'lower', 'upper', 'initial'])


def transcribe_problem(problem_data: CasadiCodesignProblemData,
                       options: CodesignSolverOptions, p_guess):

    equality_constraints = []   # all constraints == 0
    inequality_constraints = [] # all constraints >= 0

    times, colloc_coeff, diff_coeff, quad_coeff = get_collocation_matrices(
        options.degree
    )

    param_factory = ParameterFactory(problem_data.parameters)
    q = 0
    t = 0
    p = param_factory(t)

    x0 = problem_data.initial_conditions(p)
    t_grid = np.linspace(0, options.final_time, options.grid_size + 1)
    state_factory = StateFactory(problem_data, t_grid, p_guess)

    x, z = state_factory.new_terminal(0)
    equality_constraints.append(x - x0)
    y = problem_data.outputs(t, x, z, p)
    dt = options.final_time / options.grid_size
    t_out = [t]
    y_out = [y]
    q_out = [q]
    equality_constraints.append(problem_data.algebraic_constraint(t, x, z, p))
    inequality_constraints.append(problem_data.path_constraints(t, y, q, p))

    for k in range(options.grid_size):
        x_jk, z_jk = state_factory.new_collocation_points(k, options.degree)

        x_next = diff_coeff[0] * x
        for j in range(1, options.degree + 1):
            # Todo: Check if this indexing is right as C[:, 0] is never used
            dx = colloc_coeff[0, j] * x
            dx += sum(
                c_ij * x_ij
                for c_ij, x_ij in zip(colloc_coeff[1:, j], x_jk)
            )
            t_j = t + times[j - 1] * dt
            p_j = param_factory(t_j)
            args = [t_j, x_jk[j - 1], z_jk[j - 1], p_j]
            f_inter = problem_data.vector_field(*args)
            res_inter = problem_data.algebraic_constraint(*args)
            equality_constraints.append(res_inter)
            y_j = problem_data.outputs(*args)
            equality_constraints.append(dt * f_inter - dx)

            q_dot = problem_data.quadrature(t_j, y_j, p_j)
            q += quad_coeff[j] * q_dot * dt
            x_next = x_next + diff_coeff[j] * x_jk[j - 1]

        t += dt

        x, z = state_factory.new_terminal(k + 1)

        p = param_factory(t)
        equality_constraints.append(x - x_next)
        equality_constraints.append(
            problem_data.algebraic_constraint(t, x, z, p)
        )
        y = problem_data.outputs(t, x, z, p)

        c_j = problem_data.path_constraints(t, y, q, p)
        inequality_constraints.append(c_j)

        t_out.append(t)
        y_out.append(y)
        q_out.append(q)

    cost = problem_data.cost_function(t, y, q, p)
    inequality_constraints.append(problem_data.terminal_constraints(t, y, q, p))
    x_array, x_min, x_max, x_initial = state_factory.finalise()
    p_array, p_min, p_max, p_initial = param_factory.finalise()
    cost += param_factory.regularisation_cost()
    nlp_vars = casadi.vertcat(p_array, x_array)
    nlp_min = casadi.vertcat(p_min, x_min)
    nlp_max = casadi.vertcat(p_max, x_max)
    nlp_init = casadi.vertcat(p_initial, x_initial)

    eq = casadi.vertcat(*equality_constraints)
    ineq = casadi.vertcat(*inequality_constraints)
    c = casadi.vertcat(eq, ineq)
    c_min = np.zeros(c.shape, dtype=float)
    if ineq.shape != (0, 0):
        c_max = np.concatenate([np.zeros(eq.shape), np.inf * np.ones(ineq.shape)])
    else:
        c_max = np.zeros(c.shape)

    nlp_spec = {
        'f': cost,
        'x': nlp_vars,
        'g': c
    }

    nlp_args = {
        'x0': nlp_init,
        'lbx': nlp_min,
        'ubx': nlp_max,
        'lbg': c_min,
        'ubg': c_max,
        'lam_x0': np.zeros(nlp_init.shape),
        'lam_g0': np.zeros(c_max.shape)
    }
    # opts = {'ipopt.hessian_approximation':'limited-memory'}
    opts = {}
    solver = casadi.nlpsol('solver', 'ipopt', nlp_spec, opts)

    sol_to_min_and_argmin = casadi.Function(
        'argmim',
        [nlp_vars],
        param_factory.output_list()
    )

    t_out = casadi.horzcat(*t_out)
    y_out = casadi.horzcat(*y_out)
    q_out = casadi.horzcat(*q_out)

    sol_to_path = casadi.Function(
        'trajectory',
        [nlp_vars],
        [t_out, y_out, q_out]
    )

    return solver, nlp_args, sol_to_min_and_argmin, sol_to_path


class FixedTimeCodesignProblem:
    def __init__(self, problem):
        self.data, self.options = build_fixed_endpoint_codesign_problem(problem)

    def __call__(self, *args):
        """Evaluates the constrained functional with the given parameters.

        Args: list of values for decision variables."""

        # set cost = |param - args|^2
        # solve codesign problem
        # evaluate cost function
        # return result

    def minimise(self, initial_values) -> CodesignSolution:
        """Minimise the constrained function to sovle the codesign problem"""
        # set cost - problem.cost
        #

        solver, solver_args, prob, path = transcribe_problem(
            self.data,
            self.options,
            initial_values
        )

        result = solver(**solver_args)
        value = result['f'].full()
        if len(initial_values) > 1:
            argmin = [arg.full() for arg in prob(result['x'])]
        else:
            argmin = prob(result['x']).full()
        t, y, q = [a.full() for a in path(result['x'])]

        return CodesignSolution(
            cost=value,
            argmin=argmin,
            t=t,
            y=y,
            q=q
        )

