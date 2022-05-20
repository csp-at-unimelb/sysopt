"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Iterable, Callable, Optional, List
from dataclasses import dataclass, field
from functools import cached_property
from sysopt.types import Domain
from sysopt.symbolic import Variable, ExpressionGraph
from sysopt.blocks.block_operations import (
    VectorFunctionWrapper, BlockFunctionWrapper, to_graph
)

from sysopt.blocks.block_operations import create_functions_from_block
from sysopt.helpers import flatten


@dataclass
class FlattenedSystem:
    initial_conditions: Optional[ExpressionGraph] = None
    vector_field: Optional[ExpressionGraph] = None
    output_map: Optional[ExpressionGraph] = None
    constraints: Optional[ExpressionGraph] = None
    inverse_tables: Optional[dict] = None
    domain: Domain = None

    @staticmethod
    def from_block(block):

        *functions, tables = create_functions_from_block(block)
        for f in functions[1:]:
            if f:
                domain = f.domain
                break

        assert isinstance(domain, Domain)

        graphs = [to_graph(f) if f else None for f in functions]

        return FlattenedSystem(*graphs, tables, domain=domain)


@dataclass
class Quadratures:
    output_variable: Variable
    vector_quadrature: ExpressionGraph
    regularisers: List[Variable] = field(default_factory=list)
