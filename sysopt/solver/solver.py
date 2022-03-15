from typing import Optional, Dict

from sysopt import symbolic
from sysopt.solver.autodiff_context import ADContext
from sysopt.block import Composite
from sysopt.optimisation import DecisionVariable, Minimise

class TimePoint:
    is_symbolic = True

    def __new__(cls, context, value):
        if not symbolic.is_symbolic(value):
            assert isinstance(value, (float, int))
            obj = symbolic.scalar_constant(value)
        else:
            obj = value
        assert not hasattr(value, 'context')
        setattr(obj, 'context', context)
        return obj


class SolverContext:
    def __init__(self,
                 model:Composite,
                 t_final,
                 constants: Optional[Dict]=None
                 ):
        self.model = model
        self.context = ADContext(t_final)
        self.start = TimePoint(self, 0)
        self.end = TimePoint(self, t_final)
        self.t = TimePoint(self, self.context.t)
        self.constants = constants

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def evaluate(self, problem:Minimise,
                 decision_variables:Dict[DecisionVariable, float]):
        # step 1: map **parameters-> to model params

        system = self.context.get_flattened_system(self.model)
        t_final = self.context.t_final

        # step 2: generate parameter set
        parameters = self.constants.copy()
        for dv in decision_variables:
            if dv is self.context.t_final:
                t_final = float(decision_variables[self.context.t_final])
            else:
                block, slc = dv.parameter
                values = decision_variables[dv]

                if slc.stop - slc.start == 1:
                    parameters[block.parameters[slc.start]] = float(values)
                else:
                    for i in range(slc.start, slc.stop,
                                   slc.step if slc.step else 1
                                   ):
                        parameters[block.parameters[i]] = float(values[i])

        assert not symbolic.is_symbolic(t_final), "Must specify a final time"


        # step 3: add any quadratures to flattened system
        func = symbolic.Integrator(
            self.context.t,
            system
        )

        y = func(t_final, [parameters[p] for p in self.model.parameters])


        S = []

        for s, t, pi in self.context._expressions:
            if t is self.context.t_final:
                S.append(y.apply(pi)(t_final))


        cost = symbolic.evaluate(
            problem.cost, [decision_variables, ]

        )

        assert False


        return


    def solve(self, problem):

        pass

    def signal(self, parent, indices, t):

        vector = self.context.get_or_create_port_variables(parent)

        matrix = symbolic.projection_matrix(
            list(enumerate(indices)), len(vector)
        )

        return self.context.evaluate(matrix, t)

    def _get_parameter_vector(self):
        return [self.constants[p] for p in self.model.parameters]

    def integrate(self, parameters=None, resolution=50):
        if not parameters:
            parameters = self._get_parameter_vector()

        func = symbolic.Integrator(
            self.context.t,
            self.context.get_flattened_system(self.model),
            resolution=resolution
        )
        return func(self.context.t_final, parameters)

    def get_integrator(self, resolution=50):
        return symbolic.Integrator(
            self.context.t,
            self.context.get_flattened_system(self.model),
            resolution=resolution
        )