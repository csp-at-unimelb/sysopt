"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Optional, List, Union, Callable, Dict, Tuple, Iterable,  NewType
from dataclasses import dataclass, field
from collections import namedtuple, deque

from sysopt.types import Domain
from sysopt.block import Block, Composite, Connection, Channel, Port
from sysopt.symbolic import (
    Variable, ExpressionGraph, concatenate, Function, symbolic_vector,
    get_time_variable, as_function, function_from_graph,
    restriction_map
)

from sysopt.blocks.block_operations import (
    to_graph, create_functions_from_block,
    TableEntry, create_tables_from_blocks,
    get_projections_for_block
)

from sysopt import exceptions

@dataclass
class WireEntry:
    source_port: str
    source_channel: int
    destination_port: str
    destination_channel: str
    source_index: int
    destination_index: int

Arguments = namedtuple('Arguments', ['t', 'x', 'z', 'u', 'p'])
Tables = Dict[str, Union[List[TableEntry], List[WireEntry]]]


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


def is_internal(wire: Connection):
    src, dest = wire
    return src.block.parent == dest.block.parent


def get_ports_and_indices(wire: Connection) -> Iterable[Tuple[Channel, Channel]]:
    src, dest = wire

    index_pairing = list(zip(src.indices, dest.indices))
    src_port = src.port if isinstance(src, Channel) else src
    dest_port = dest.port if isinstance(dest, Channel) else dest

    return src_port, dest_port, index_pairing


def find_channel_in_table(table: List[TableEntry],
                          port: Port,
                          local_index: int):

    def key(entry: TableEntry):
        return (entry.block == str(port.block)
                and entry.local_index == local_index)

    entry, = list(filter(key, table))

    return entry.global_index


def internal_wire_to_table_entries(tables: Tables,
                                   wire: Connection) -> List[WireEntry]:
    # for the source
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        src_index = find_channel_in_table(tables['outputs'], src_port, src_i)
        dest_index = find_channel_in_table(tables['inputs'], dest_port, dest_i)
        entries.append(WireEntry(
            source_port=str(src_port),
            destination_port=str(dest_port),
            source_channel=src_i,
            destination_channel=dest_i,
            source_index=src_index,
            destination_index=dest_index
        ))
    return entries

def is_wire_forwarding_inputs(wire):
    src, dest = wire
    return src.port_type == dest.port_type == 'inputs'


def forwarded_input_to_table_entry(tables:Tables, wire: Connection):
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        global_index = find_channel_in_table(tables['inputs'], dest_port, dest_i)
        entries.append(TableEntry(
            local_name=src_port.channel_name(src_i),
            block=str(src_port.block),
            local_index=src_i,
            global_index=global_index
        ))
    return entries

def forwarded_output_to_table_entry(tables: Tables, wire: Connection):
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        global_index = find_channel_in_table(tables['outputs'], src_port, src_i)
        entries.append(TableEntry(
            local_name=dest_port.channel_name(dest_i),
            block=str(dest_port.block),
            local_index=dest_i,
            global_index=global_index
        ))
    return entries


def create_tables(all_blocks:List[Block]):
    tables = create_tables_from_blocks(
        *filter(lambda b: not isinstance(b, Composite), all_blocks)
    )

    trunks = list(filter(lambda b: isinstance(b, Composite), all_blocks))
    domain = Domain(
        1,
        len(tables['states']),
        len(tables['constraints']),
        len(tables['inputs']),
        len(tables['parameters'])
    )

    tables['wires'] = []
    while trunks:
        next_block: Composite = trunks.pop()
        for wire in next_block.wires:
            if is_internal(wire):
                tables['wires'] += internal_wire_to_table_entries(tables, wire)
            elif is_wire_forwarding_inputs(wire):
                entries = forwarded_input_to_table_entry(tables, wire)
                tables['inputs'] += entries
            else:
                entries = forwarded_output_to_table_entry(tables, wire)
                tables['outputs'] += entries

    return tables, domain

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



def create_symbols_from_domain(domain):
    arguments = Arguments(
        t=get_time_variable(),
        x=symbolic_vector('states', domain.states),
        z=symbolic_vector('constraints', domain.constraints),
        u=symbolic_vector('inputs', domain.inputs),
        p=symbolic_vector('parameters', domain.parameters)
    )

    return arguments


def symbolically_evaluate_initial_conditions(block:Block,
                                             local_arguments: Arguments):
    try:
        x0 = block.initial_state(local_arguments.p)
    except NotImplementedError as ex:
        raise exceptions.FunctionError(
            block, block.initial_state, 'function is not implemented!'
        ) from ex
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
    return x0

def symbolically_evaluate(block: Block,
                          func: Callable,
                          dimension: int,
                          local_arguments: Arguments):

    try:
        f = func(*local_arguments)
    except NotImplementedError as ex:
        raise exceptions.FunctionError(
            block, func, 'function is not implemented!') from ex
    except Exception as ex:
        raise exceptions.FunctionError(
            block, func, ex.args
        ) from ex
    f = concatenate(*f)
    if f.shape != (dimension, ):
        raise exceptions.FunctionError(
            block,func,
            f'Expected shape {(dimension, )} but ' \
            f'the function returned a vector of shape {f.shape}'
        )

    return f

def symbolically_evaluate_block(tables: Dict,
                                block: Block,
                                arguments: Arguments):
    proj = get_projections_for_block(tables, block)
    proj_x = proj['states']
    proj_z = proj['constraints']
    proj_y = proj['outputs']
    local_args = Arguments(
        t=arguments.t,
        x=proj_x @ arguments.x,
        z=proj_z @ arguments.z,
        u=proj['inputs'] @ arguments.u,
        p=proj['parameters'] @ arguments.p
    )

    x0 = proj_x.T @ symbolically_evaluate_initial_conditions(
        block, local_args
    ) if block.signature.states > 0 else None

    f = proj_x.T @ symbolically_evaluate(
            block, block.compute_dynamics, block.signature.states, local_args
    )  if block.signature.states > 0 else None

    g = proj_y.T @ symbolically_evaluate(
        block, block.compute_outputs, block.signature.outputs, local_args
    )

    h = proj_z.T @ symbolically_evaluate(
        block, block.compute_residuals, block.signature.constraints, local_args
    ) if block.signature.constraints > 0 else None

    return x0, f, g, h


def create_constraints_from_wire_list(
    wires: List[WireEntry],
    arguments: Arguments,
    outputs: ExpressionGraph) -> ExpressionGraph:

    sources, sinks = zip(
        *((wire.source_index, wire.destination_index) for wire in wires)
    )
    proj_u = restriction_map(indices=list(sinks),
                             superset_dimension=len(arguments.u))
    proj_y = restriction_map(indices=list(sources),
                             superset_dimension=outputs.shape[0])
    vector_constraint = proj_u(arguments.u) - proj_y (outputs)

    return vector_constraint


def flatten_system(root: Composite):
    all_blocks = tree_to_list(root)
    leaves = filter(lambda x: not isinstance(x,Composite), all_blocks)
    tables, domain = create_tables(all_blocks)
    symbols = create_symbols_from_domain(domain)

    function_lists = zip(*[
        symbolically_evaluate_block(tables, block, symbols)
        for block in leaves
    ])
    initial_conditions, vector_field, output_map, constraints = [
        sum([f for f in function_list if f is not None])
        for function_list in function_lists
    ]
    wiring_constraint = create_constraints_from_wire_list(
        tables['wires'], symbols, output_map
    )
    constraints = concatenate(constraints, wiring_constraint)

    return FlattenedSystem(
        initial_conditions=initial_conditions,
        vector_field=vector_field,
        output_map=output_map,
        constraints=constraints,
        domain=domain,
        inverse_tables=tables
    )


