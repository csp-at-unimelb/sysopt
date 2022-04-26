"""Methods and objects for solving system optimisation problems."""

import dataclasses
import weakref
from typing import Optional, Dict, List, Union, Iterable, NewType

from sysopt import symbolic
from sysopt.symbolic import (
    ExpressionGraph, Variable, Parameter, get_time_variable
)

from sysopt.solver.symbol_database import SymbolDatabase
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
        self.symbol_db = SymbolDatabase(t_final)
        self.start = 0
        self.end = t_final
        self.t = get_time_variable()
        self.constants = constants
        self.resolution = path_resolution

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_symbolic_integrator(self,
                                decision_vars: Iterable[DecisionVariable]):
        integrator = self.get_integrator()
        parameter_arguments = {}

        for dv in decision_vars:
            if dv is not self.symbol_db.t_final:
                block, slc = dv.parameter

                if slc.stop - slc.start == 1:

                    parameter_arguments[block.parameters[slc.start]] = dv
                else:
                    iterator = range(
                        slc.start, slc.stop, slc.step if slc.step else 1
                    )
                    for i in iterator:
                        parameter_arguments[block.parameters[i]] = dv[i]
        parameters = symbolic.concatenate(
            *[parameter_arguments[p] if p in parameter_arguments
              else self.constants[p] for p in self.model.parameters]
        )

        symbolic_evaluation = integrator(self.t, parameters)

        f = symbolic.lambdify(
            symbolic_evaluation, [self.t,
                                  symbolic.concatenate(decision_vars)]
        )
        return f

    def _prepare_path(self, decision_variables:Dict[DecisionVariable, float]):
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

        if not parameters:
            parameters = self._get_parameter_vector()

        integrator = self.get_integrator(resolution)
        if not t_final:
            t_final = self.symbol_db.t_final
        soln = integrator.integrate(t_final, parameters)

        return soln

    @property
    def flattened_system(self):
        return self.symbol_db.get_flattened_system(self.model)

    def get_integrator(self, resolution=50):
        return symbolic.Integrator(
            self.symbol_db.t,
            self.flattened_system,
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

    def problem(self, cost, arguments, subject_to=None):
        return Problem(self, cost, arguments, subject_to)


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
                 cost: ExpressionGraph,
                 arguments: List[Variable],
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
