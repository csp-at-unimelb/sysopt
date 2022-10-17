"""Module for Optimisation-Based Controllers."""

from typing import List, Union

from functools import lru_cache

from sysopt import Block, Metadata
from sysopt.backends import get_variational_solver
from sysopt.symbolic import Function
from sysopt.symbolic.symbols import (
    SymbolicArray, Variable
)
from sysopt.problems.problem_data import MinimumPathProblem, SolverOptions


from sysopt.helpers import flatten


def get_names_of_symbolic_atoms(
        arg: Union[List[SymbolicArray], SymbolicArray]) -> List[str]:

    if isinstance(arg, list):
        return flatten([get_names_of_symbolic_atoms(a) for a in arg])

    return [f'{arg}_{i}' for i in range(len(arg))]


class PathPlanner(Block):
    """System Component for Optimial Path Planning.

    Args:
        problem: Description of path planning problem
        name: component name.

    """
    def __init__(self,
                 problem: MinimumPathProblem,
                 solver_options: SolverOptions = None,
                 name=None):

        if problem.parameters:
            param_names = ['T'] + get_names_of_symbolic_atoms(
                problem.parameters)
        else:
            param_names = ['T']

        metadata = Metadata(
            outputs=get_names_of_symbolic_atoms(
                [problem.state[0], problem.control[0]]
            ),
            parameters=param_names
        )
        super().__init__(metadata, name)
        self._problem = problem
        self._solver_options = solver_options or SolverOptions()

        @lru_cache(1)
        def solver(t_final):
            return get_variational_solver(self._problem,
                                          self._solver_options)(t_final)

        def func(t, p):
            soln = solver(p[0])(p[1:])
            return soln(t)

        self._solver = Function(
            arguments=[Variable('t'), Variable('p', self.signature.parameters)],
            function=func,
            shape=problem.state[0].shape
        )

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return self._solver(t, parameters)
