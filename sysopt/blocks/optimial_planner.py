from typing import List, Union

from sysopt import Block, Metadata
from sysopt.backends import get_variational_solver
from sysopt.symbolic import Function
from sysopt.symbolic.symbols import (
    SymbolicArray, MinimumPathProblem, Variable
)


from sysopt.helpers import flatten


def get_names_of_symbolic_atoms(
        arg: Union[List[SymbolicArray], SymbolicArray]) -> List[str]:

    if isinstance(arg, list):
        return flatten([get_names_of_symbolic_atoms(a) for a in arg])

    return [f'{arg}_{i}' for i in range(len(arg))]


class PathPlanner(Block):
    def __init__(self, problem: MinimumPathProblem, name=None):

        if problem.parameters:
            param_names = ['T'] + get_names_of_symbolic_atoms(
                problem.parameters)
        else:
            param_names = ['T']

        metadata = Metadata(
            outputs=get_names_of_symbolic_atoms([problem.state[0],
                                                 problem.control[0]]),
            parameters=param_names
        )
        super().__init__(metadata, name)
        self._problem = problem
        solver = get_variational_solver(self._problem)

        def func(t, p):
            soln = solver(p[0])(p[1:])
            return soln(t)

        self._solver = Function(
            arguments=[Variable('t'), Variable('p', self.metadata.parameters)],
            function=func,
            shape=problem.state[0].shape
        )

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return self._solver(t, parameters)