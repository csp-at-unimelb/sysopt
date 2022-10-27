"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name

from typing import Optional, List, Union, Callable, Dict, Tuple

from dataclasses import dataclass, asdict
from collections import deque
from sysopt.problems.problem_data import Domain
from sysopt.modelling.block import (
    Block, Composite, ComponentBase, DiscreteBlock
)
from sysopt.symbolic import (
    Variable, ExpressionGraph, concatenate, symbolic_vector,
    get_time_variable, function_from_graph,
    restriction_map, as_array, sparse_matrix, Matrix
)
from sysopt.problems.problem_data import FlattenedSystem
from sysopt.problems.wiring_tables import (
    TableEntry, Tables, global_dimension_of, WireEntry, create_tables
)

from sysopt import exceptions


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


def projection_from_entries(entries: List[TableEntry],
                            global_dim: int,
                            local_dim: int) -> Matrix:
    shape = (local_dim, global_dim)
    m = sparse_matrix(shape)
    for entry in entries:
        row = entry.local_index
        col = entry.global_index
        m[row, col] = 1
    return m


def inclusion_from_entries(entries: List[TableEntry],
                           global_dim: int,
                           local_dim: int) -> Matrix:
    proj = projection_from_entries(entries, global_dim, local_dim)
    return proj.T


def get_projections_for_block(tables: Tables, block: Block):
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


def symbolically_evaluate_initial_conditions(block: Block,
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
        raise exceptions.FunctionError(block, func, message) from ex

    except Exception as ex:
        message = f'An execption \'{ex}\' was raised during ' \
                  'symbolic evaluation'

        raise exceptions.FunctionError(block, func, message) from ex

    f = as_array(f)
    if f.shape != (dimension, ):
        raise exceptions.FunctionError(
            block, func,
            f'Expected shape {(dimension, )} but '
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

    discrete = filter(lambda x: isinstance(x, DiscreteBlock), all_blocks)
    cts = filter(
        lambda x: not isinstance(x, (DiscreteBlock, Composite)), all_blocks
    )

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

    if not output_indices:
        raise exceptions.NoTopLevelOutputs(root)

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
