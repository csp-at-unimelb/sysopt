"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
import copy
from typing import Optional, List, Union, Callable, Dict, Tuple, Iterable
from dataclasses import dataclass, asdict
from collections import deque

from sysopt.types import Domain
from sysopt.block import (
    Block, Composite, Connection, Channel, Port, ComponentBase, DiscreteBlock
)
from sysopt.symbolic import (
    Variable, ExpressionGraph, concatenate, symbolic_vector,
    get_time_variable, function_from_graph,
    restriction_map, as_array, sparse_matrix, Matrix
)
from sysopt.symbolic.problem_data import FlattenedSystem


from sysopt import exceptions


@dataclass
class TableEntry:
    """Name and index of a block variable."""
    local_name: str
    block: str
    local_index: int
    global_index: int

    @property
    def name(self):
        return f'{self.block}/{self.local_name}'

    def __str__(self):
        return f'\"{self.name}\", {self.global_index} -> {self.local_index})\n'

    def __repr__(self):
        return f'{self.global_index} -> ({self.name}, {self.local_index})\n'


@dataclass
class WireEntry:
    source_port: str
    source_channel: int
    destination_port: str
    destination_channel: str
    source_index: int
    destination_index: int


Tables = Dict[str, Union[List[TableEntry], List[WireEntry]]]


@dataclass
class Arguments:
    """Container for symbolic function arguments"""
    t: Variable
    x: Union[Variable, ExpressionGraph]
    z: Union[Variable, ExpressionGraph]
    u: Union[Variable, ExpressionGraph]
    p: Union[Variable, ExpressionGraph]

    @property
    def domain(self) -> Domain:
        return Domain(
            1, *[v.shape[0] if v is not None else 0
                 for v in (self.x, self.z, self.u, self.p)]
        )

    def __iter__(self):
        return iter((self.t, self.x, self.z, self.u, self.p))



def is_internal(wire: Connection) -> bool:
    src, dest = wire
    return src.block.parent == dest.block.parent


def get_ports_and_indices(wire: Connection) \
    -> Iterable[Tuple[Channel, Channel]]:

    src, dest = wire

    index_pairing = list(zip(src.indices, dest.indices))
    src_port = src.port if isinstance(src, Channel) else src
    dest_port = dest.port if isinstance(dest, Channel) else dest

    return src_port, dest_port, index_pairing


def projection_from_entries(entries: List[TableEntry],
                            global_dim:int,
                            local_dim: int) -> Matrix:
    shape = (local_dim, global_dim)
    m = sparse_matrix(shape)
    for entry in entries:
        row = entry.local_index
        col = entry.global_index
        m[row, col] = 1
    return m


def inclusion_from_entries(entries:List[TableEntry],
                           global_dim: int,
                           local_dim: int) -> Matrix:
    proj = projection_from_entries(entries, global_dim, local_dim)
    return proj.T


def create_tables_from_block(block: Block) -> Tables:

    tables = {}

    for attribute, attr_object in asdict(block.metadata).items():
        if attr_object:
            tables[attribute] = [
                TableEntry(block=str(block), local_name=name,
                           local_index=i, global_index=i)
                for i, name in enumerate(attr_object)
            ]
        else:
            tables[attribute] = []

    return tables


def merge_table(table_1: Tables, table_2: Tables) -> Tables:
    out_table = {}
    for key in set(table_1.keys()) | set(table_2.keys()):
        l1 = table_1[key] if key in table_1 else []
        l2 = table_2[key] if key in table_2 else []
        out_table[key] = [copy.copy(entry) for entry in l1]
        offset = global_dimension_of(l1)

        out_table[key] += [
            TableEntry(local_name=entry.local_name,
                       block=entry.block,
                       local_index=entry.local_index,
                       global_index=entry.global_index + offset)
            for entry in l2
        ]
    return out_table


def create_tables_from_blocks(*blocks):

    tables = [create_tables_from_block(block) for block in blocks]

    base, *rest = tables

    for table in rest:
        base = merge_table(base, table)

    return base

def global_dimension_of(table):
    if not table:
        return 0
    return 1 + max(entry.global_index for entry in table)

def get_projections_for_block(tables:Tables , block: Block):
    projectors = {}
    for attr, local_dim in asdict(block.signature).items():
        if local_dim == 0:
            projectors[attr] = None
            continue
        entries = sorted([
            entry for entry in tables[attr] if entry.block == str(block)
        ], key=lambda entry: entry.local_index)
        projectors[attr] = projection_from_entries(
            entries,
            local_dim=local_dim,
            global_dim=global_dimension_of(tables[attr])
        )

    return projectors


def find_channel_in_table(table: List[TableEntry],
                          port: Port,
                          local_index: int) -> int:

    def key(entry: TableEntry):
        return (entry.block == str(port.block)
                and entry.local_index == local_index)

    try:
        entry, = list(filter(key, table))
    except ValueError as ex:
        if len(list(filter(key, table))) == 0:
            message = f'Coulnd not find {port.block}' \
                      f'[{local_index}] in table\n'
        else:
            message = f'Multiple entries for {port.block}' \
                      f'[{local_index}] in table: {table}\n'
        raise ValueError(message) from ex

    return entry.global_index


def internal_wire_to_table_entries(tables: Tables,
                                   wire: Connection) -> List[WireEntry]:
    # for the source
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        try:
            src_index = find_channel_in_table(
                tables['outputs'], src_port, src_i
            )
            dest_index = find_channel_in_table(
                tables['inputs'], dest_port, dest_i
            )
        except ValueError as ex:
            message = f'Failed to get ports for wire {wire} : {ex.args}'
            raise ValueError(message) from ex
        entries.append(WireEntry(
            source_port=str(src_port),
            destination_port=str(dest_port),
            source_channel=src_i,
            destination_channel=dest_i,
            source_index=src_index,
            destination_index=dest_index
        ))
    return entries


def is_wire_forwarding_inputs(wire: Connection) -> bool:
    src, dest = wire
    return src.port_type == dest.port_type == 'inputs'


def forwarded_input_to_table_entry(tables:Tables,
                                   wire: Connection) -> List[TableEntry]:
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        global_index = find_channel_in_table(tables['inputs'],
                                             dest_port,
                                             dest_i)

        entries.append(TableEntry(
            local_name=src_port.channel_name(src_i),
            block=str(src_port.block),
            local_index=src_i,
            global_index=global_index
        ))
    return entries

def forwarded_output_to_table_entry(tables: Tables,
                                    wire: Connection) -> List[TableEntry]:
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


def create_tables(all_blocks:List[Block]) -> Tables:
    if isinstance(all_blocks, Composite):
        return create_tables(tree_to_list(all_blocks))

    tables = create_tables_from_blocks(
        *filter(lambda b: not isinstance(b, Composite), all_blocks)
    )

    trunks = list(filter(lambda b: isinstance(b, Composite), all_blocks))
    sizes = [
        global_dimension_of(tables[name])
        for name in ('states', 'constraints', 'inputs', 'parameters')
    ]

    domain = Domain(1, *sizes)

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

def tree_to_list(block: Union[Composite, Block]) -> List[ComponentBase]:
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



def create_symbols_from_domain(domain) -> Arguments:
    arguments = Arguments(
        t=get_time_variable(),
        x=symbolic_vector('states', domain.states),
        z=symbolic_vector('constraints', domain.constraints),
        u=symbolic_vector('inputs', domain.inputs),
        p=symbolic_vector('parameters', domain.parameters)
    )

    return arguments


def symbolically_evaluate_initial_conditions(block:Block,
                                             local_arguments: Arguments
                                             ) -> ExpressionGraph:

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

    x0 = as_array(x0)

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
    except exceptions.InvalidShape as ex:
        domain = local_arguments.domain
        message = 'Failed to evaluate symbolically when called with '\
                  f'local arguments of {domain}'
        raise exceptions.FunctionError(
            block, func, message
        ) from ex
    except Exception as ex:
        message = f'An execption \'{ex}\' was raised during ' \
                  'symbolic evaluation'
        raise exceptions.FunctionError(
            block, func, message
        ) from ex
    f = as_array(f)
    if f.shape != (dimension, ):
        raise exceptions.FunctionError(
            block, func,
            f'Expected shape {(dimension, )} but ' \
            f'the function returned a vector of shape {f.shape}'
        )

    return f


def get_local_args_for_block(proj, block, arguments):
    proj_x = proj['states']
    proj_z = proj['constraints']
    proj_p = proj['parameters']
    proj_u = proj['inputs']
    local_args = Arguments(
        t=arguments.t,
        x=proj_x @ arguments.x if block.signature.states else None,
        z=proj_z @ arguments.z if block.signature.constraints else None,
        u=proj_u @ arguments.u if block.signature.inputs else None,
        p=proj_p @ arguments.p if block.signature.parameters else None
    )
    return local_args


def symbolically_evaluate_continuous_block(
    tables: Dict, block: Block, arguments: Arguments
) -> Tuple[Optional[ExpressionGraph]]:
    proj = get_projections_for_block(tables, block)
    proj_x = proj['states']
    proj_z = proj['constraints']
    proj_y = proj['outputs']
    local_args = get_local_args_for_block(proj, block, arguments)

    x0 = proj_x.T @ symbolically_evaluate_initial_conditions(
        block, local_args
    ) if block.signature.states else None

    f = proj_x.T @ symbolically_evaluate(
            block, block.compute_dynamics, block.signature.states, local_args
    ) if block.signature.states else None

    g = proj_y.T @ symbolically_evaluate(
        block, block.compute_outputs, block.signature.outputs, local_args
    )

    h = proj_z.T @ symbolically_evaluate(
        block, block.compute_residuals, block.signature.constraints, local_args
    ) if block.signature.constraints > 0 else None

    return x0, f, g, h


def symbolically_evaluate_discrete_block(
    tables: Dict, block: DiscreteBlock, arguments: Arguments
) -> Tuple[Optional[ExpressionGraph]]:

    proj = get_projections_for_block(tables, block)
    proj_x = proj['states']
    proj_z = proj['constraints']
    proj_y = proj['outputs']
    local_args = get_local_args_for_block(proj, block, arguments)

    x0 = proj_x.T @ symbolically_evaluate_initial_conditions(
        block, local_args
    ) if block.signature.states else None

    f = proj_x.T @ (symbolically_evaluate(
        block, block.compute_state_transition,
        block.signature.states, local_args
    ) - local_args.x) if block.signature.states else None

    g = proj_y.T @ symbolically_evaluate(
        block, block.compute_outputs, block.signature.outputs, local_args
    )

    h = proj_z.T @ symbolically_evaluate(
        block, block.compute_residuals, block.signature.constraints, local_args
    ) if block.signature.constraints > 0 else None

    return (x0, f, g, h), block.frequency


def create_constraints_from_wire_list(wires: List[WireEntry],
                                      arguments: Arguments,
                                      outputs: ExpressionGraph
                                      ) -> ExpressionGraph:

    sources, sinks = zip(
        *((wire.source_index, wire.destination_index) for wire in wires)
    )
    proj_u = restriction_map(indices=list(sinks),
                             superset_dimension=len(arguments.u))
    proj_y = restriction_map(indices=list(sources),
                             superset_dimension=outputs.shape[0])

    vector_constraint = proj_u(arguments.u) - proj_y(outputs)

    return vector_constraint


def flatten_system(root: ComponentBase) -> FlattenedSystem:
    all_blocks = tree_to_list(root)

    tables, domain = create_tables(all_blocks)
    symbols = create_symbols_from_domain(domain)
    discrete = filter(lambda x: isinstance(x, DiscreteBlock),
                      all_blocks)
    cts = filter(lambda x: not isinstance(x, (DiscreteBlock, Composite)),
                 all_blocks)

    # Flatten all continuous blocks
    function_lists = zip(*[
        symbolically_evaluate_continuous_block(tables, block, symbols)
        for block in cts
    ])

    function_lists = [
        list(filter(lambda x: x is not None, function_list))
        for function_list in function_lists
    ]

    initial_conditions, vector_field, output_map, constraints = [
        sum(function_list) if function_list else None
        for function_list in function_lists
    ]
    state_transitions = []
    # Flatten discrete blocks

    discrete_funcs = [
        symbolically_evaluate_discrete_block(tables, block, symbols)
        for block in discrete
    ]
    for (x0, f, g, h), freq in discrete_funcs:

        initial_conditions += x0
        state_transitions.append(
            (freq,
             function_from_graph(f, symbols),
             function_from_graph(h, symbols) if h else None))

        output_map += g

    if tables['wires']:
        wiring_constraint = create_constraints_from_wire_list(
            tables['wires'], symbols, output_map
        )
        if constraints:
            constraints = concatenate(constraints, wiring_constraint)
        else:
            constraints = wiring_constraint

    output_tables = filter(
        lambda entry: entry.block == str(root),
        tables['outputs']
    )

    output_indices = {
        entry.local_index: entry.global_index for entry in output_tables
    }

    proj_y = restriction_map(output_indices, output_map.shape[0])
    outs = proj_y(output_map)
    return FlattenedSystem(
        initial_conditions=function_from_graph(initial_conditions,
                                               [symbols.p]),
        vector_field=function_from_graph(vector_field, symbols),
        output_map=function_from_graph(outs, symbols),
        constraints=function_from_graph(constraints, symbols),
        state_transitions=state_transitions if state_transitions else None,
        domain=domain,
        tables=tables
    )


