"""Sympy implementation for generating a vector field from system model."""

from typing import Optional
from sysopt.problems.problem_data import (
    FlattenedSystem, Quadratures, Domain
)
from sysopt.backends.sympy.expression_graph import sympy_vector
from sysopt.backends import get_backend
backend = get_backend('sympy')


def get_integrator(system: FlattenedSystem,
                   *args,
                   quadratures: Optional[Quadratures] = None,
                   **kwargs
                   ):

    return SympyDAE(system, quadratures)


def symbols_from_domain(domain: Domain):
    return [
        sympy_vector(name, (dim,)) if dim > 0 else None
        for name, dim in zip(domain.letters(), domain)
    ]


class SympyDAE:
    """Sympy representation of the system model.

    Args:
        system:         The flattened system model.
        quadratures:    Feedback-less integrals of outs to be evaluated.

    Attributes:
        vector_field: The equation 'f' that generates the dynamics
        constraints:  System Algebraic constraints
        output_map:   The system outputs

    """

    def __init__(self,
                 system: FlattenedSystem,
                 quadratures: Optional[Quadratures] = None):

        self.symbols = symbols_from_domain(system.domain)

        f = backend.get_implementation(system.vector_field)
        self.vector_field = f(*self.symbols)

        self.constraints = None
        if system.constraints:
            h = backend.get_implementation(system.constraints)
            self.constraints = h(*self.symbols)

        g = backend.get_implementation(system.output_map)
        self.output_map = g(*self.symbols)

        x0 = backend.get_implementation(system.initial_conditions)
        self.initial_conditions = x0(self.symbols[-1])

        self.quadratures = None

    def __call__(self, t, params):
        subs = [(self.symbols[-1], params)]
        out = {
            'f': self.vector_field.subs(subs),
            'g': self.output_map.subs(subs),
            'x0': self.initial_conditions.subs(subs)
        }
        if self.constraints:
            out['h'] = self.constraints.subs(subs)

        return out
