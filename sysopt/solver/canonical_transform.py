"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Optional, List, Union,Callable ,Dict
from dataclasses import dataclass, field
from collections import namedtuple, deque

from sysopt.types import Domain
from sysopt.block import Block, Composite
from sysopt.symbolic import Variable, ExpressionGraph, concatenate,Function
from sysopt.blocks.block_operations import (
    to_graph, create_functions_from_block,
    TableEntry, create_tables_from_blocks
)

from sysopt import exceptions

Arguments = namedtuple('Arguments', ['t', 'x', 'z', 'u', 'p'])

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



def tree_to_list(block: Union[Composite, Block]):
    fifo = deque()
    fifo.append(block)
    result = []
    while fifo:
        item = fifo.popleft()
        result.append(item)
        try:
            for component in item.components:
                fifo.append(component)
        except AttributeError:
            pass
    return result

def create_leaf_tables(block_list: List[Union[Composite, Block]]):

    tables = create_tables_from_blocks(
        *filter(lambda b: isinstance(b, Block), block_list)
    )

    return tables



def symbolically_evaluate_initial_conditions(block:Block,
                                             arguments:Arguments):
    try:
        x0 = block.initial_state(arguments.p)
    except NotImplementedError:
        raise exceptions.FunctionError(
            block, block.initial_state, 'function is not implemented!')
    except Exception as ex:
        raise exceptions.FunctionError(
            block, block.initial_state, ex.args
        ) from ex
    x0 = concatenate(*x0)
    expected_shape = (block.signature.states,)
    if x0.shape != expected_shape:
        raise exceptions.FunctionError(
            block, block.initial_state,
            f'Expected shape {expected_shape} but ' \
            f'the function returned {x0.shape}'
        )
    return Function.from_graph(x0, list(arguments.p.symbols()))

def symbolically_evaluate(block: Block,
                          func: Callable,
                          dimension: int,
                          arguments: Arguments):

    try:
        f = func(*arguments)
    except Exception as ex:
        raise exceptions.FunctionError(
            block, func, ex.args
        )
    f = concatenate(*f)
    if f.shape != (dimension, ):
        raise exceptions.FunctionError(
            block,func,
            f'Expected shape {(dimension, )} but ' \
            f'the function returned a vector of shape {f.shape}'
        )
    return Function.from_graph(
        f, [a for arg in arguments for a in arg.symbols()]
    )

def wrap_block_functions(tables: Dict, block: Block, arguments: Arguments):
    proj = get_projections_for_block(tables, block)
    proj_x = proj['states']
    proj_z = proj['constraints']
    proj_y = proj['outputs']
    local_args = Arguments(
        t=arguments.t,
        x=proj_x @ arguments.x,
        z=proj_z@ arguments.z,
        u=proj['inputs'] @ arguments.u,
        p=proj['parameters'] @ arguments.p
    )
    x0 = symbolic_evaluate_initial_conditions(block, arguments)

    f = proj_x.T @ symbolically_evaluate(
        block, block.compute_dynamics, block.signature.states
    )
    g = proj_y @ symbolically_evaluate(
        block, block.compute_outputs, block.signature.outputs
    )
    h = symbolically_evaluate(
        block, block.compute_residuals, block.signature.constraints
    )
    return x0, f, g, h