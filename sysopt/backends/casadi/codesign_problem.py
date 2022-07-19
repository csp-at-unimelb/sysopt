#pylint: skip-file

import casadi
from sysopt.backends.casadi.function import compiles, to_function
from sysopt.backends.casadi.expression_graph import CasadiGraphWrapper
from sysopt.symbolic.problem_data import ConstrainedFunctional


def build_function(problem: ConstrainedFunctional):
    p = casadi.vertcat(*[casadi.SX.sym(p_i) for p_i in problem.parameters])

    value = to_function(problem.value)
    if isinstance(problem.final_time, float):
        t_f = casadi.SX(problem.final_time)
    else:
        t_f = to_function(problem.final_time)(p)

    args = to_function(problem.parameter_map)(p)


@compiles(ConstrainedFunctional)
class CodesignProblem:
    def __init__(self, problem: ConstrainedFunctional):
        self.func = build_function(problem)

    def __call__(self, *args):
        pass

    def pushforward(self, *args):
        pass



