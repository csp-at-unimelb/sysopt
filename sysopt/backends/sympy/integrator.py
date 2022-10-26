from typing import Optional
from sysopt.problems.problem_data import FlattenedSystem, Quadratures
from sysopt.backends.sympy.expression_graph import compile_expression_graph


def get_integrator(system:FlattenedSystem,
                   resolution=50,
                   quadratures: Optional[Quadratures]=None):

    return SympyDAE(system, quadratures)


class SympyDAE:
    def __init__(self,
                 system: FlattenedSystem,
                 quadratures: Optional[Quadratures] = None):

        self.vector_field = compile_expression_graph(system.vector_field)
        self.constraints = compile_expression_graph(system.constraints)
        self.output = compile_expression_graph(system.output_map)
        print(system.tables)

    def __call__(self, *args, **kwargs):
        pass