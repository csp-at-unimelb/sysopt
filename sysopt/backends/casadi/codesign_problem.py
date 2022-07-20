#pylint: skip-file

import casadi
from sysopt.backends.casadi.compiler import implements, get_implementation
from sysopt.backends.casadi.expression_graph import CasadiGraphWrapper
from sysopt.symbolic.problem_data import ConstrainedFunctional


def build_function(problem: ConstrainedFunctional):
    p = casadi.vertcat(*[casadi.SX.sym(p_i) for p_i in problem.parameters])

    value = get_implementation(problem.value)
    if isinstance(problem.final_time, float):
        t_f = casadi.SX(problem.final_time)
    else:
        t_f = get_implementation(problem.final_time)(p)

    args = get_implementation(problem.parameter_map)(p)


@implements(ConstrainedFunctional)
class CodesignProblem:
    def __init__(self, problem: ConstrainedFunctional):
        self.func = build_function(problem)

    def __call__(self, *args):
        pass

    def pushforward(self, *args):
        pass



