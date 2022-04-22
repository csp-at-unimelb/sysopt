"""Methods and objects for solving system optimisation problems."""

import dataclasses
import weakref
from typing import Optional, Dict, List, Union, NewType

from sysopt import symbolic
from sysopt.symbolic import (
    ExpressionGraph, Variable, Parameter, get_time_variable,
    lambdify
)

from sysopt.solver.symbol_database import SymbolDatabase
from sysopt.block import Block, Composite
import sysopt.backends as backend
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
        self.symbol_db = SymbolDatabase(t_final)
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
        self._flat_system = self.symbol_db.get_flattened_system(self.model)
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
        pass

    def add_quadrature(self, integrand):

        y_var = self.model.outputs(self.t)
        unbound_params = integrand.symbols() - {y_var, self.t}
        assert not unbound_params

        f = lambdify(integrand, [self.t, y_var])
        dot_q = f(self.symbol_db.t, self._flat_system.g)
        idx = self._flat_system.add_quadrature(dot_q)
        return idx

    def get_symbolic_integrator(self):

        integrator = self.get_integrator()
        p = symbolic.SymbolicVector('p', len(self.parameters))

        p_symbols = self._parameter_map(p)
        symbolic_evaluation = integrator(self.symbol_db.t, p_symbols)

        f = backend.lambdify(
            symbolic_evaluation, [self.symbol_db.t, p]
        )
        return f

    def _prepare_path(self, decision_variables: Dict[DecisionVariable, float]):
        t_final = self.symbol_db.t_final
        parameters = self.constants.copy()

        for dv in decision_variables:
            if dv is self.symbol_db.t_final:
                t_final = float(decision_variables[self.symbol_db.t_final])
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

    def evaluate(self, problem: 'Problem',
                 decision_variables: Dict[DecisionVariable, float]):

        y, _ = self._prepare_path(decision_variables)

        t = get_time_variable()
        y_vars = self.model.outputs(t)
        arguments = {y_vars: y}.update(decision_variables)

        value = problem.cost.call(arguments)
        constraints = []
        return CandidateSolution(value, y, constraints)

    def solve(self, problem):
        pass

    def _get_parameter_vector(self):
        return [self.constants[p] for p in self.model.parameters]

    def integrate(self, parameters=None, t_final=None, resolution=50):

        integrator = self.get_integrator(resolution)
        p = self._parameter_map(parameters)
        if not t_final:
            t_final = self.symbol_db.t_final

        soln = integrator.integrate(t_final, p)

        return soln

    @property
    def flattened_system(self):
        return self._flat_system

    def get_integrator(self, resolution=50):
        assert self._flat_system is not None
        return symbolic.Integrator(
            self.symbol_db.t,
            self._flat_system,
            resolution=resolution
        )

    def is_time_varying(self, symbol_or_expression):

        symbols = symbolic.list_symbols(symbol_or_expression)
        if self.t in symbols:
            return True

        for s, _ in self.symbol_db.path_variables:
            for s_prime in symbols:
                if id(s) == id(s_prime):
                    return True

        return False

    def problem(self, arguments, cost, subject_to=None):
        return Problem(self, arguments, cost, subject_to)


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

    @property
    def context(self):
        return self._context()

    @property
    def cost(self):
        return self._cost

    def __call__(self, *args):
        """Evaluate the problem with the given arguments."""
        assert len(args) == len(self.arguments), \
            f'Invalid arguments: expected {self.arguments}, received {args}'
        arg_dict = dict(zip(self.arguments, args))

        return self.context.evaluate(self, arg_dict)

