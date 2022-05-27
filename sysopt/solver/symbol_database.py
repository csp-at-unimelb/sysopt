"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Optional, List
from dataclasses import dataclass, field
from sysopt.types import Domain
from sysopt.blocks import Block
from sysopt.symbolic import Variable, ExpressionGraph
from sysopt.blocks.block_operations import (
    to_graph, create_functions_from_block
)


@dataclass
class FlattenedSystem:
    """Container for flattened system functions."""
    initial_conditions: Optional[ExpressionGraph] = None
    vector_field: Optional[ExpressionGraph] = None
    output_map: Optional[ExpressionGraph] = None
    constraints: Optional[ExpressionGraph] = None
    inverse_tables: Optional[dict] = None
    domain: Domain = None

    @staticmethod
    def from_block(block: Block):
        """Creates a new flattened system from the given block"""
        domain = None
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
    """Container for quadratures associated with a given system."""
    output_variable: Variable
    vector_quadrature: ExpressionGraph
    regularisers: List[Variable] = field(default_factory=list)
