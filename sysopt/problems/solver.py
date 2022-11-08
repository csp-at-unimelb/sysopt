"""Methods and objects for solving system optimisation problems."""

import weakref
from typing import Optional, Dict, List, Union, NewType

import numpy as np

from sysopt import symbolic
from sysopt.backends import get_backend
from sysopt.symbolic import (
    ExpressionGraph, Variable, get_time_variable,
    is_symbolic, ConstantFunction, GraphWrapper, PiecewiseConstantSignal
)
from sysopt.problems.canonical_transform import flatten_system
from sysopt.problems.problem_data import Quadratures, ConstrainedFunctional, FlattenedSystem, CollocationSolverOptions
from sysopt.modelling.block import Block, Composite
from sysopt.exceptions import InvalidParameterException

DecisionVariable = NewType(
    'DecisionVariable',
    Union[Variable, PiecewiseConstantSignal])


class SolverContext:
    """Numerical/algorithmic solver context manager.

    Args:
        model:      System model under treatment
        t_final:    Final time for simulations/optimisations to be valid
        constants:  A mapping from model parameters to numbers, setting the
                    constant numerical values for this run.
        backend:    The symbolic backend; currently `casadi` or `sympy`

    Examples:
        See `README.md` for example usage.

    """
    def __init__(self,
                 model: Union[Block, Composite],
                 t_final: Union[float, Variable],
                 parameters: Optional[Dict] = None,
                 backend='casadi'
                 ):
        self.model = model
        self.start = 0
        self.t_final = t_final
        self.t = get_time_variable()

        self._flat_system = flatten_system(self.model)

        self.quadratures = None

        # fill in missing parameters with symbols
        parameters = {
            name: Variable(name)
            if not parameters or name not in parameters else parameters[name]
            for name in model.parameters
        }

        self.parameter_map = None
        self.symbols, self.parameters, t_map, p_map = create_parameter_map(
            self.model, parameters, self.t_final
        )
        self.parameter_map = p_map
        self.t_final_map = t_map
        self._params_to_t_final = t_map
        self.__ctx = get_backend(backend)

    def __enter__(self):
        self.__ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__ctx.__exit__(exc_type, exc_val, exc_tb)

    def integral(self, integrand):
        return symbolic.Quadrature(integrand, self)

    def add_quadrature(self, integrand):
        if not self.quadratures:
            idx = 0
            self.quadratures = Quadratures(
                self.model.outputs(self.t), integrand)
        else:
            idx = self.quadratures.vector_quadrature.shape[0]
            self.quadratures.vector_quadrature = symbolic.concatenate(
                self.quadratures.vector_quadrature,
                integrand
            )
        return idx

    def get_implementation(self, obj):
        return self.__ctx.get_implementation(obj)

    @property
    def flattened_system(self) -> FlattenedSystem:
        return self._flat_system

    def problem(self, arguments, cost, subject_to=None):
        return Problem(self, arguments, cost, subject_to)

    def get_symbolic_integrator(self):
        integrator = self.get_integrator()
        param_map = self.__ctx.get_implementation(self.parameter_map)
        time_map = self.__ctx.get_implementation(self.t_final_map)

        def f(p):
            t_f = time_map(p)
            args = param_map(p)
            return integrator(t_f, args)
        return f

    def evaluate_quadrature(self, index, t, params):
        integrator = self.get_integrator()
        args = self.parameter_map(params)
        _, q = integrator(t, args)

        return q[index]

    def integrate(self, parameters=None, t_final=None, resolution=50):

        integrator = self.get_integrator(resolution)
        try:
            p = self.parameter_map(parameters)
        except (ValueError, TypeError) as ex:
            raise InvalidParameterException(
                f'Failed to map parameters arguments \'{parameters}\' '
                f'to {self.parameters}.'
            ) from ex

        if not t_final:
            t_final = self.t_final_map(parameters)

        soln = integrator.integrate(t_final, p)

        return soln

    def get_integrator(self, resolution=50):
        return self.__ctx.get_integrator(
            self._flat_system,
            resolution=resolution,
            quadratures=self.quadratures
        )

    @property
    def backend(self):
        return self.__ctx


def lambdify_terminal_constraint(problem: 'Problem',
                                 constraint: symbolic.Inequality):
    t_f = problem.context.t_final
    terminal_values = problem.context.model.outputs(t_f)
    args = [terminal_values, problem.arguments]

    return symbolic.function_from_graph(constraint.to_graph(), args)


class Problem:
    """Optimisation Problem.

    Args:
        context:        Model context for this problem.
        cost:           Symbolic expression for cost function.
        arguments:      Decision variables/arguments for cost.
        constraints:    Path, terminal and parameter constraints for the
            problem.

    """

    def __init__(self,
                 context: SolverContext,
                 arguments: List[Variable],
                 cost: ExpressionGraph,
                 constraints: Optional[List[ExpressionGraph]]):
        # cost function is split into the form
        # J(T) = f(y_T, p) + q(T, p)
        # where f is the terminal form
        # and q is the quadrature; such that \dot{q} = g(y,t,p)
        # q(0) = 0 so that q = \int_0^T g\df{t}

        self._context = weakref.ref(context)
        self.arguments = arguments
        """Symbolic variables matching the unbound parameters"""
        self.constraints: List[symbolic.Inequality] = constraints or []
        self._cost = cost

    @property
    def context(self):
        return self._context()

    @property
    def cost(self):
        return self._cost

    def _get_problem_specification(self):
        context = self.context

        t = get_time_variable()
        y = self.context.model.outputs(context.t)
        param_args, = self.context.parameter_map.arguments
        t_final = self.context.t_final_map

        symbols = {
            self.arguments[i]: param_args[i]
            for i, p_i in enumerate(self.context.parameters)
        }
        terminal_cost, q, dot_q = symbolic.extract_quadratures(self.cost)
        terminal_cost = symbolic.replace_signal(
            terminal_cost, y, context.t_final, y
        )

        terminal_cost = symbolic.substitute(terminal_cost, symbols)

        if dot_q:
            dot_q = GraphWrapper(symbolic.substitute(dot_q, symbols),
                                 [t, y, param_args])

        if q is not None:
            cost_args = [t, y, q, param_args]
        else:
            cost_args = [t, y,
                         symbolic.symbolic_vector('q', 0),
                         param_args]

        cost_fn = symbolic.function_from_graph(terminal_cost, cost_args)
        parameters = {
            a: [-np.inf, np.inf] for a in self.arguments
        }

        path_constraints = []
        point_constraints = []
        for constraint in self.constraints:
            if is_box_constraint(constraint, parameters.keys()):
                smaller = constraint.smaller
                bigger = constraint.bigger
                if smaller in parameters:
                    old_max = parameters[smaller][1]
                    parameters[smaller][1] = min(old_max, bigger)
                    continue
                elif bigger in parameters:
                    old_min = parameters[bigger][0]
                    parameters[bigger][0] = max(old_min, smaller)
                    continue
            if symbolic.is_temporal(constraint):
                c = symbolic.substitute(constraint.to_graph(), symbols)
                path_constraints.append(GraphWrapper(c, cost_args))
            else:
                c = symbolic.replace_signal(
                    constraint.to_graph(),
                    y, context.t_final, y
                )

                c = symbolic.substitute(c, symbols)

                point_constraints.append(GraphWrapper(c, cost_args))

        spec = ConstrainedFunctional(
            final_time=t_final,
            parameter_map=self.context.parameter_map,
            system=self.context.flattened_system,
            quadratures=dot_q,
            value=cost_fn,
            parameters=parameters,
            point_constraints=point_constraints,
            path_constraints=path_constraints
        )
        return spec

    def __call__(self, args):
        """Evaluate the problem with the given arguments."""
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'

        spec = self._get_problem_specification()

        integrator = self.context.get_integrator()

        t = self.context.t_final_map(args)
        p = self.context.parameter_map(args)

        backend = get_backend()
        if self.context.quadratures:
            y, q = integrator(t, p)
            q = backend.as_array(q)
        else:
            y = integrator(t, p)
            q = None

        y = backend.as_array(y)
        cost = spec.value(t, y, q, args)

        return cost

    def jacobian(self, args):
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'
        spec = self._get_problem_specification()
        # create a functional object
        # with
        # - vector field
        # - constraints
        # - initial conditions
        # -
        integrator = self.context.get_integrator()
        cost = self.context.get_implementation(spec.value)
        n = len(self.arguments)
        t = self.context.t_final
        jac = np.zeros((n, 1), dtype=float)
        for i in range(n):
            basis = np.array([0 if i != j else 1 for j in range(n)])
            y, q, dy, dq = integrator.pushforward(
                t, args, basis)

            _, dcost = cost.pushforward(t, y, q, args, 0, dy, dq, basis)
            jac[i] = dcost
        return jac

    def solve(self, guess, options:Optional[CollocationSolverOptions]=None):

        problem = self._get_problem_specification()
        solver = self.context.get_implementation(problem)
        opts = self.context.get_implementation(
            options or CollocationSolverOptions()
        )

        return solver.minimise(guess, opts)


def create_parameter_map(model, constants, final_time):

    basis_map = {}
    param_constants = np.zeros((len(model.parameters), ), dtype=float)
    params = []
    if is_symbolic(final_time):
        params.append(final_time)

    for idx, name in enumerate(model.parameters):
        try:
            value = constants[name]
        except KeyError as ex:
            message = f'Model parameter {ex.args[0]} not specified! '
            raise ValueError(message) from ex
        if is_symbolic(value):
            try:
                param_idx = params.index(value)
            except ValueError:
                param_idx = len(params)
                params.append(value)
            basis_map[idx] = param_idx
        else:
            param_constants[idx] = value

    # Parameter Map should look like:
    #
    # t_final = < (e_0, params) >
    # p_final = [ b_i (e^i , params) ,...]
    # where e^i is the cobasis vector of the parameter in the domain (inputs)
    # and b_i is the corresponding basis in the output space (output, index)

    if is_symbolic(final_time):
        args = symbolic.symbolic_vector('parameters', len(params))
        pi = symbolic.inclusion_map({0: 0}, len(params), 1)
        t_func = GraphWrapper(pi(args), [args])
    else:
        args = symbolic.symbolic_vector('parameters', len(params))
        t_func = ConstantFunction(final_time, arguments=args)

    if basis_map:
        m = symbolic.sparse_matrix((len(model.parameters), len(params)))
        for row, col in basis_map.items():
            m[row, col] = 1

        expr = m @ args
        p_func = GraphWrapper(expr + param_constants, [args])
    else:
        p_func = ConstantFunction(param_constants, args)
    return args, params, t_func, p_func


def is_box_constraint(constraint: symbolic.Inequality, symbols):
    try:
        return ((constraint.smaller in symbols
                 and not is_symbolic(constraint.bigger))
                or (constraint.bigger in symbols
                    and not is_symbolic(constraint.smaller)))
    except AttributeError:
        return False
