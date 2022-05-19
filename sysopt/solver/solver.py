"""Methods and objects for solving system optimisation problems."""

import dataclasses
import weakref
from typing import Optional, Dict, List, Union, NewType

from sysopt import symbolic
from sysopt.backends import Integrator, lambdify
from sysopt.symbolic import (
    ExpressionGraph, Variable, Parameter, get_time_variable,
    is_temporal, create_log_barrier_function, as_vector, is_symbolic
)

from sysopt.solver.symbol_database import FlattenedSystem
from sysopt.block import Block, Composite

DecisionVariable = NewType('DecisionVariable', Union[Variable, Parameter])


@dataclasses.dataclass
class CandidateSolution:
    """A candidate solution to a constrained optimisation problem. """
    cost: float
    trajectory: object
    constraints: Optional[List[float]] = None


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
        self.end = t_final
        self.t = get_time_variable()
        self.constants = constants
        self.resolution = path_resolution
        self.quadratures = []
        self.parameters = [t_final] if isinstance(t_final, Variable) else []
        self._flat_system = None
        self._parameter_map = None

    def __enter__(self):
        self._flat_system = FlattenedSystem.from_block(self.model)
        try:
            free_params = list(
                set(self.model.parameters) - set(self.constants.keys())
            )
        except AttributeError:
            free_params = self.model.parameters

        for param in free_params:
            block, index = self.model.find_by_type_and_name('parameters', param)
            self.parameters.append(Parameter(block, index))

        self._parameter_map = self.get_parameter_map()
        return self

    def get_parameter_map(self):

        if self.constants and \
                set(self.constants.keys()) == set(self.model.parameters):
            return lambda _: list(self.constants.values())

        result = []
        index = 0
        constants = self.constants if self.constants else []
        for param in self.model.parameters:
            if param not in constants:
                result.append(self.parameters[index])
                index += 1
            else:
                result.append(self.constants[param])
        result = symbolic.concatenate(*result)

        return symbolic.lambdify(result, self.parameters)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flat_system = None

    # def add_quadrature(self, integrand):
    #
    #     y_var = self.model.outputs(self.t)
    #     unbound_params = integrand.symbols() - {y_var, self.t}
    #     assert not unbound_params
    #
    #     #f = symbolic.lambdify(integrand, [self.t, y_var])
    #     args = [
    #         self.t,
    #     ]
    #     expression = integrand.call(
    #
    #     )
    #     dot_q = f(self.end, *self._flat_system.g)
    #     idx = self._flat_system.add_quadrature(dot_q)
    #     return idx

    def prepare_path(self, decision_variables: Dict[DecisionVariable, float]):
        t_final = self.end
        parameters = self.constants.copy()

        for dv in decision_variables:
            if dv is self.end:
                t_final = float(decision_variables[self.end])
            else:
                block, slc = dv.get_source_and_slice()
                values = decision_variables[dv]

                if slc.stop - slc.start == 1:
                    parameters[block.parameters[slc.start]] = float(values)
                else:
                    iterator = range(
                        slc.start, slc.stop, slc.step if slc.step else 1
                    )
                    for i in iterator:
                        parameters[block.parameters[i]] = float(values[i])

        assert not symbolic.is_symbolic(t_final), 'Must specify a final time'
        integrator = self.get_integrator(self.resolution)

        params = [float(parameters[p]) for p in self.model.parameters]
        func = integrator.integrate(t_final, params)

        return func, t_final

    def _create_parameter_projections(self):
        constants = []
        proj_indices = []

        if symbolic.is_symbolic(self.end):
            constants.append(0)
            proj_indices.append(1)
        else:
            constants.append(self.end)

        for row, parameter in enumerate(self.model.parameters):
            try:
                constants.append(self.constants[parameter])
            except KeyError:
                constants.append(0)
                proj_indices.append(row)

        out_dimension = len(constants)
        arguments = symbolic.Variable(shape=(len(proj_indices),))
        projector = symbolic.projection_matrix(proj_indices, out_dimension).T
        const_vector = symbolic.array(constants)
        graph = projector @ arguments + const_vector

        return symbolic.lambdify(graph, [arguments], 'params')

    def get_symbolic_integrator(self):
        integrator = self.get_integrator()
        param_map = self._create_parameter_projections()

        if is_symbolic(self.end):
            def f(t, p):
                args = param_map(p)
                return integrator(t, args)
        else:
            def f(p):
                args = param_map(p)
                return integrator(args[0], args[1:])
        return f

        # if symbolic.is_symbolic(self.end):
        #     def func(t, p):
        #         p_prime = param_map.call(as_vector(p))
        #         return integrator(t, *p_prime)
        # else:
        #     def func(p):
        #         p_prime = param_map.call(as_vector(p))
        #         t_end = self.end
        #         return integrator(t_end, *p_prime)
        # return func

    def evaluate(self, problem: 'Problem',
                 decision_variables: Dict[DecisionVariable, float]):
        y, _ = self.prepare_path(decision_variables)

        t = get_time_variable()
        y_vars = self.model.outputs(t)
        arguments = {y_vars: y}
        arguments.update(decision_variables)

        value = problem.cost.call(arguments)
        constraints = []
        for constraint in problem.constraints:
            if is_temporal(constraint):
                raise NotImplementedError
            else:
                g = constraint.to_graph()
                constraints.append(g.call(arguments))

        return CandidateSolution(value, y, constraints)

    def solve(self, problem):
        raise NotImplementedError

    # def _get_parameter_vector(self):
    #     return [self.constants[p] for p in self.model.parameters]

    def integrate(self, parameters=None, t_final=None, resolution=50):

        integrator = self.get_integrator(resolution)

        p = self._parameter_map(parameters)

        if not t_final:
            t_final = self.end
        soln = integrator.integrate(t_final, p)

        return soln

    @property
    def flattened_system(self):
        return self._flat_system

    def get_integrator(self, resolution=50):
        return Integrator(
            self._flat_system,
            resolution=resolution
        )

    def problem(self, arguments, cost, subject_to=None):
        return Problem(self, arguments, cost, subject_to)


def lambdify_terminal_constraint(problem: 'Problem',
                                 constraint: symbolic.Inequality):
    t_f = problem.context.end
    terminal_values = problem.context.model.outputs(t_f)
    args = [terminal_values, problem.arguments]

    return symbolic.lambdify(constraint.to_graph(), args)


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
        self._cost = cost
        self._context = weakref.ref(context)
        self.arguments = arguments
        self.constraints = constraints if constraints else []
        self._regularisers = []

    @property
    def context(self):
        return self._context()

    @property
    def cost(self):
        return self._cost

    def __call__(self, args):
        """Evaluate the problem with the given arguments."""
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'
        arg_dict = dict(zip(self.arguments, args))
        return self.call(arg_dict)

    def call(self, args):

        y, _ = self.context.prepare_path(args)

        t = get_time_variable()
        y_vars = self.context.model.outputs(t)
        arguments = {y_vars: y}
        arguments.update(args)
        cost_term, cost_quad = symbolic.extract_quadratures(self.cost)
        rho = Variable('rho')
        assert not cost_quad, 'Not implemented'
        constraints = []
        for constraint in self.constraints:
            if is_temporal(constraint):
                raise NotImplementedError
            else:
                g = constraint.to_graph()
                constraints.append(g.call(arguments))
                barrier = create_log_barrier_function(g, rho)
                cost_term += barrier

        if constraints:
            arguments.update({rho: 0.001})

        value = cost_term.call(arguments)
        return CandidateSolution(value, y, constraints)

