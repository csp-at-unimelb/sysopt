"""Methods and objects for solving system optimisation problems."""

import weakref
from typing import Optional, Dict, List, Union, NewType

import numpy as np

from sysopt import symbolic
from sysopt.backends import get_integrator, get_implementation
from sysopt.symbolic import (
    ExpressionGraph, Variable, Parameter, get_time_variable,
    is_symbolic, ConstantFunction, GraphWrapper
)
from sysopt.solver.canonical_transform import flatten_system
from sysopt.symbolic.problem_data import Quadratures, ConstrainedFunctional, FlattenedSystem
from sysopt.block import Block, Composite

DecisionVariable = NewType('DecisionVariable', Union[Variable, Parameter])


class InvalidParameterException(Exception):
    pass


class SolverContext:
    """Context manager for model simulation and optimisation.

    Args:
        model:  The model block diagram

    """
    def __init__(self,
                 model: Union[Block, Composite],
                 t_final: Union[float, Variable],
                 constants: Optional[Dict] = None,
                 path_resolution: int = 50
                 ):
        self.model = model
        self.start = 0
        self.t_final = t_final
        self.t = get_time_variable()
        self.constants = constants if constants else {}
        self.resolution = path_resolution
        self.quadratures = None
        self._flat_system = None
        self.parameter_map = None
        self._params_to_t_final = None
        self.parameters = None

    def __enter__(self):
        self._flat_system = flatten_system(self.model)
        _, self.parameters, t_map, p_map = create_parameter_map(
            self.model, self.constants, self.t_final
        )
        self.parameter_map = p_map
        self._params_to_t_final = t_map

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flat_system = None

    def prepare_path(self, decision_variables: Dict[DecisionVariable, float]):
        try:
            values = [
                decision_variables[p] for p in self.parameters
            ]
        except KeyError as ex:

            raise ValueError(
                f'Undefined parameters: expected {self.parameters}, '
                f'received {decision_variables}') from ex

        t_final = self._params_to_t_final(values)
        params = self.parameter_map(values)
        integrator = self.get_integrator()
        func = integrator.integrate(t_final, params)

        return func, t_final

    def _create_parameter_projections(self):
        constants = []
        proj_indices = []

        if symbolic.is_symbolic(self.t_final):
            constants.append(0)
            proj_indices.append(1)
        else:
            constants.append(self.t_final)
        free_params = []
        for row, parameter in enumerate(self.model.parameters):
            try:
                constants.append(self.constants[parameter])
            except KeyError:
                constants.append(0)
                proj_indices.append(row)
                free_params.append(parameter)
        out_dimension = len(constants)
        arguments = symbolic.Variable(
            shape=(len(proj_indices), ),
            name=f'''[{','.join(free_params)}]''')
        basis_map = dict(enumerate(proj_indices))
        projector = symbolic.inclusion_map(
            basis_map, len(proj_indices), out_dimension
        )
        const_vector = symbolic.array(constants)
        pi_args = projector(arguments)

        graph = pi_args + const_vector
        return symbolic.function_from_graph(graph, [arguments])

    def get_symbolic_integrator(self):
        integrator = self.get_integrator()
        param_map = self._create_parameter_projections()

        if is_symbolic(self.t_final):
            def f(t, p):
                args = param_map(p)
                return integrator(t, args)
        else:
            def f(p):
                args = param_map(p)
                return integrator(args[0], args[1:])
        return f

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
            t_final = self.t_final
        soln = integrator.integrate(t_final, p)

        return soln

    @property
    def flattened_system(self) -> FlattenedSystem:
        return self._flat_system

    def get_integrator(self, resolution=50):
        return get_integrator(
            self._flat_system,
            resolution=resolution,
            quadratures=self.quadratures
        )

    def problem(self, arguments, cost, subject_to=None):
        return Problem(self, arguments, cost, subject_to)

    def integral(self, integrand):
        return symbolic.Quadrature(integrand, self)


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

        self._impl = None
        self._terminal_cost = None
        self._cost = cost

    @property
    def context(self):
        return self._context()

    @property
    def cost(self):
        return self._cost

    def _get_problem_specification(self):
        context = self.context
        param_args, parameters, t_final, param_map = create_parameter_map(
            self.context.model, self.context.constants, self.context.t_final,
        )
        t = get_time_variable()
        y = self.context.model.outputs(context.t)
        symbols = {
            self.arguments[i]: param_args[i]
            for i, p_i in enumerate(parameters)
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
            parameter_map=param_map,
            system=self.context.flattened_system,
            quadratures=dot_q,
            value=cost_fn,
            parameters=parameters,
            point_constraints=point_constraints,
            path_constraints=path_constraints
        )
        self._terminal_cost = cost_fn
        return spec

    def __call__(self, args):
        """Evaluate the problem with the given arguments."""
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'
        spec = self._get_problem_specification()

        _ = get_implementation(spec)

        integrator = self.context.get_integrator()

        t = self.context._params_to_t_final(args)
        p = self.context.parameter_map(args)

        if self.context.quadratures:
            y, q = integrator(t, p)
        else:
            y = integrator(t, p)
            q = None
        cost = self._terminal_cost(t, y, q, args)

        return cost

    def jacobian(self, args):
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'
        _ = self._get_problem_specification()
        # create a functional object
        # with
        # - vector field
        # - constraints
        # - initial conditions
        # -
        integrator = self.context.get_integrator()
        n = len(self.arguments)
        t = self.context.t_final
        jac = np.zeros((n, 1), dtype=float)
        for i in range(n):
            basis = np.array([0 if i != j else 1 for j in range(n)])
            y, q, dy, dq = integrator.pushforward(
                t, args, basis)
            _, dcost = self._terminal_cost.pushforward(t, y, q, args,
                                                       0, dy, dq, basis)
            jac[i] = dcost
        return jac

    def solve(self, initial_values):

        problem = self._get_problem_specification()
        solver = get_implementation(problem)
        return solver.minimise(initial_values)


def create_parameter_map(model, constants, final_time):
    try:
        output_idx, params = zip(*[
            (idx, Parameter(model, name))
            for idx, name in enumerate(model.parameters)
            if not constants or name not in constants
        ])
        params = list(params)
    except ValueError:
        output_idx = []
        params = []
    # Parameter Map should look like:
    #
    # t_final = < (e_0, params) >
    # p_final = [ b_i (e^i , params) ,...]
    # where e^i is the cobasis vector of the parameter in the domain (inputs)
    # and b_i is the correspnding basis in the output space (output, index)
    offset = 0
    param_constants = np.array([
        constants[p] if p in constants else 0 for p in model.parameters
    ]) if constants else np.zeros((len(model.parameters), ), dtype=float)

    if is_symbolic(final_time):
        params.insert(0, final_time)
        args = symbolic.symbolic_vector('parameter vector', len(params))
        pi = symbolic.inclusion_map({0: 0}, len(params), 1)
        t_func = GraphWrapper(pi(args), [args])
        offset = 1
    else:
        args = symbolic.symbolic_vector('parameter vector', len(params))
        t_func = ConstantFunction(final_time, arguments=args)

    if output_idx:
        basis_map = {
                in_i + offset: out_i for in_i, out_i in enumerate(output_idx)
        }
        inject = symbolic.inclusion_map(
            basis_map,
            len(params),
            len(model.parameters)
        )

        p_func = GraphWrapper(inject(args) + param_constants, [args])
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
