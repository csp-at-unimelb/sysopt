"""Interface for Symbolic Functions and AutoDiff."""

import copy
from dataclasses import dataclass, asdict
from typing import Callable, Union, Iterable


from sysopt.types import Domain
from sysopt.block import Block, Composite, check_wiring_or_raise
from sysopt.helpers import flatten, strip_nones
from sysopt.symbolic.symbols import (
    as_vector, sparse_matrix, as_function, symbolic_vector,
    get_time_variable
)
from sysopt.symbolic.function_ops import (
    FunctionOp, Concatenate, project, compose,
)


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

    def __repr__(self):
        return f'\"{self.name}\", ({self.global_index}, {self.local_index})\n'


def projection_from_entries(entries, global_dim, local_dim):
    shape = (local_dim, global_dim)
    m = sparse_matrix(shape)
    for entry in entries:
        row = entry.local_index
        col = entry.global_index
        m[row, col] = 1
    return m


def inclusion_from_entries(entries, global_dim, local_dim):
    proj = projection_from_entries(entries, global_dim, local_dim)
    return proj.T


def coproduct(domain, *functions: 'FunctionOp'):
    r"""Returns the functional coproduct of arguments

    For :math:`f_i:\oplus_j X^i_j ->Y_i`, this method
    returns a function :math:`F: \oplus_j (\oplus_iX^_j) -> \oplus_i Y_i`

    That is, the domain of the $j$th $X_j$ argument of $F$ is the direct sum
    (over $i$) of the $j$th arguments of $f_i$, $X_j = \oplus_i X^i_j$.

    When calling the returned function, the stacked input vectors are
    split into their target domains, then passed to the respective functions.

    Args:
        domain:     The domain of the composed funciton
        *functions: FunctionOps with the same number of arguments.

    Returns: Operation that computes the function coproduct.

    """
    klasses = {f.__class__ for f in functions}
    if all(issubclass(k, FunctionOp) for k in klasses):
        return BlockFunctionWrapper.coproduct(domain, functions)
    if all(issubclass(k, VectorFunctionWrapper) for k in klasses):
        return VectorFunctionWrapper.coproduct(domain, functions)

    raise NotImplementedError


class VectorFunctionWrapper:
    """Wrapper for a function from $R^n$ to $R^m$.

    Args:
        domain: Dimension of the input vector
        codomain: Dimension of the output vector
        function: Function to compute the result.

    """

    def __init__(self,
                 domain: int,
                 codomain:  int,
                 function: Callable
                 ):
        self.domain = domain
        self.codomain = codomain
        self.function = function

    def __call__(self, *args):
        return self.function(*args)

    @staticmethod
    def coproduct(domain, args: Iterable['VectorFunctionWrapper']):
        d = 0
        codomain = 0
        func_list = []

        for func in args:
            func_list.append(
                (func, slice(d, d + func.domain),
                 slice(codomain, codomain + func.codomain))
            )
            d += func.domain
            codomain += func.codomain

        def call(x):
            result = [0] * codomain
            for f, dom_slice, codom_slice in func_list:
                result[codom_slice] = f(x[dom_slice])
            return result

        return VectorFunctionWrapper(
            domain,
            codomain,
            call
        )


def concatenate_block_func(*args):
    flat_args = strip_nones(flatten(args, 2))
    if any(isinstance(arg, FunctionOp) for arg in flat_args):
        return Concatenate(*flat_args)
    # if any(is_symbolic(arg) for arg in flat_args):
    #     return concatenate_symbols(*flat_args)
    if all(isinstance(arg, (float, int)) for arg in flat_args):
        return flat_args

    raise NotImplementedError


def to_graph(wrapper):

    if isinstance(wrapper.domain, int):
        x = symbolic_vector('parameters', wrapper.domain)
        return wrapper.function(x)

    symbols = [
        symbolic_vector(name, length) if length > 0 else None
        for name, length in asdict(wrapper.domain).items()
    ]

    symbols[0] = get_time_variable()
    symbolic_call = wrapper(*symbols)
    graph = as_function(symbolic_call, symbols)

    return graph


class BlockFunctionWrapper(FunctionOp):
    """Wrapper for differentiable functions.

    Args:
        domain:     The dimensions of the input space.
        codomain:   The dimensions of the output space.
        function:   The function definition.

    """
    def __init__(self,
                 domain: Union[Domain],
                 codomain: int,
                 function: Callable):
        self.domain = Domain(*domain)
        self.codomain = codomain

        self.function = function

    def __call__(self, t, *args):
        args = [
            as_vector(a) if not isinstance(a, list) else a
            for a in args
        ]

        return self.function(t, *args)

    @staticmethod
    def _partition_func(d: Domain,
                        codomain: int,
                        func: 'BlockFunctionWrapper'):
        x_slice = slice(d.states, d.states + func.domain.states)
        z_slice = slice(d.constraints, d.constraints + func.domain.constraints)
        u_slice = slice(d.inputs, d.inputs + func.domain.inputs)
        p_slice = slice(d.parameters, d.parameters + func.domain.parameters)

        def f(t, x, z, u, p):
            return func(t, x[x_slice], z[z_slice], u[u_slice], p[p_slice])

        return slice(codomain, codomain + func.codomain), f

    @staticmethod
    def coproduct(domain, args: Iterable['BlockFunctionWrapper']):
        d = Domain(1, 0, 0, 0, 0)
        codomain = 0

        func_list = []
        for func in args:
            func_list.append(
                 BlockFunctionWrapper._partition_func(d, codomain, func)
            )
            d += func.domain
            codomain += func.codomain

        def call(*args):

            result = [0] * codomain

            for sl, f in func_list:
                r = as_vector(f(*args))
                result[sl] = r
            return result

        return BlockFunctionWrapper(
            domain=domain,
            codomain=codomain,
            function=call
        )


class ArgPermute(FunctionOp):
    """Operator that remaps inputs to states variables or constraints.

    Args:
        codomain: The domain of the wrapped function.

    """
    def __init__(self, codomain: Domain):

        self.codomain = Domain(*codomain)
        self.domain = Domain(*codomain)
        self.constraint_to_input = {}
        self.input_to_input = {}

    def remap_input_as_constraint(self, original_index):
        assert (original_index not in self.constraint_to_input or
                original_index not in self.input_to_input), \
            'Index already mapped'
        z_idx = self.domain.constraints
        self.domain.constraints += 1
        self.domain.inputs -= 1
        self.constraint_to_input[original_index] = z_idx
        return z_idx

    def permute_input(self, original_index, new_index):
        assert (original_index not in self.constraint_to_input or
                original_index not in self.input_to_input), \
            'Index already mapped'
        self.input_to_input[original_index] = new_index

    def __call__(self, t, x, z, u, p):
        # This is the input vector that the internal components see

        u_inner = [
            z[self.constraint_to_input[i]] if i in self.constraint_to_input
            else u[self.input_to_input[i]]
            for i in range(self.codomain.inputs)
        ]

        z_inner = z[:self.codomain.constraints]
        return t, x, z_inner, u_inner, p


def _create_functions_from_leaf_block(block: Block):
    domain = Domain(
        1,
        block.signature.states,
        block.signature.constraints,
        block.signature.inputs,
        block.signature.parameters
    )
    x0 = None
    f = None
    g = None
    h = None
    if block.signature.states > 0:
        x0 = VectorFunctionWrapper(
            domain=block.signature.parameters,
            codomain=block.signature.states,
            function=block.initial_state
        )
        f = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.states,
            function=block.compute_dynamics
        )

    if block.signature.outputs > 0:
        g = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.outputs,
            function=block.compute_outputs
        )
    if block.signature.constraints > 0:
        h = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.constraints,
            function=block.compute_residuals
        )

    tables = create_tables_from_block(block)
    return x0, f, g, h, tables


def create_tables_from_block(block):

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


def merge_table(table_1, table_2):
    out_table = {}
    for key in set(table_1.keys()) | set(table_2.keys()):
        l1 = table_1[key] if key in table_1 else []
        l2 = table_2[key] if key in table_2 else []
        out_table[key] = [copy.copy(entry) for entry in l1]
        offset = len(l1)
        out_table[key] += [
            TableEntry(local_name=entry.local_name,
                       block=entry.block,
                       local_index=entry.local_index,
                       global_index=entry.global_index + offset)
            for entry in l2
        ]
    return out_table


def create_functions_from_block(block: Union[Block, Composite]):
    try:
        functions = {component: create_functions_from_block(component)
                     for component in block.components}
        check_wiring_or_raise(block)
    except AttributeError:
        # block is a leaf block
        return _create_functions_from_leaf_block(block)
    domain_offsets = {}
    output_offsets = {}
    output_codomain = 0
    domain = Domain()
    out_table = {}

    for component, (_, f, g, h, table) in functions.items():

        domain_offsets[component] = copy.copy(domain)
        domain += g.domain
        out_table = merge_table(out_table, table)
        if g:
            output_offsets[component] = output_codomain
            output_codomain += g.codomain

    lists = zip(*list(functions.values()))
    x0_list, f_list, g_list, h_list, _ = [
        strip_nones(item) for item in lists
    ]

    h = coproduct(domain, *h_list) if h_list else None
    x0 = coproduct(domain.parameters, *x0_list) if x0_list else None
    f = coproduct(domain, *f_list) if f_list else None
    g = coproduct(domain, *g_list) if g_list else None

    if not block.wires:
        return x0, f, g, h, out_table

    arg_permute = ArgPermute(domain)
    in_wires = [
        (src, dest) for src, dest in block.wires
        if src.port_type == dest.port_type == 'inputs'
    ]
    out_wires = [
        (src, dest) for src, dest in block.wires
        if src.port_type == dest.port_type == 'outputs'
    ]

    internal_wires = [
        (i, src, dest) for i, (src, dest) in enumerate(block.wires)
        if src.port_type != dest.port_type
    ]

    for src, dest in in_wires:
        component_offset = domain_offsets[dest.parent].inputs
        for input_index, component_index in zip(src.indices, dest.indices):
            arg_permute.permute_input(
                original_index=component_offset + component_index,
                new_index=input_index
            )

    g_actual = [0] * len(block.outputs)
    for src, dest in out_wires:
        component_offset = output_offsets[src.parent]
        for component_index, external_index in zip(src.indices, dest.indices):
            proj = project(g.codomain, component_offset + component_index)
            g_actual[external_index] = compose(proj, g, arg_permute)

    new_constraints = {}
    wire_names = []
    offset = len(out_table['constraints'])
    for i, src, dest in internal_wires:
        wire_names.append(
            TableEntry(
                block=str(block),
                local_name=f'wire from {src} -> {dest}',
                local_index=i, global_index=offset + i
            )
        )
        src_offset = output_offsets[src.parent]
        dest_offset = domain_offsets[dest.parent].inputs
        for src_idx, dest_idx in zip(src.indices, dest.indices):

            z_index = arg_permute.remap_input_as_constraint(
                dest_idx + dest_offset)

            proj = project(g.codomain, src_offset + src_idx)
            # proj: output_internal -> Reals
            # g: X_internal -> output_internal
            # arg_permute: X_external -> X_internal
            new_constraints[z_index] = compose(proj, g, arg_permute)

    h_new = [
        project(arg_permute.domain, 'constraints', z_i) - g_i
        for z_i, g_i in new_constraints.items()
    ]
    out_table['constraints'] += wire_names

    f = compose(f, arg_permute)
    if h:
        h = concatenate_block_func(compose(h, arg_permute), *h_new)
    else:
        h = concatenate_block_func(*h_new)
    if g_actual:
        items = list(g_actual[i] for i in range(len(block.outputs)))
        g = concatenate_block_func(*items)
    else:
        g = None

    return x0, f, g, h, out_table


def partition_tree(block, leaves, trunks):
    try:
        for c in block.components:
            partition_tree(c, leaves, trunks)
        trunks.append(block)
    except AttributeError:
        leaves.append(block)


def create_tables_from_blocks(*blocks):

    tables = [create_tables_from_block(block) for block in blocks]

    base, *rest = tables

    for table in rest:
        base = merge_table(base, table)

    return base


def get_projections_for_block(tables, block):
    projectors = {}
    for attr, local_dim in asdict(block.signature).items():
        if local_dim == 0:
            projectors[attr] = None
            continue
        entries = sorted([
            entry for entry in tables[attr] if entry.block is block
        ], key=lambda entry: entry.local_index)
        projectors[attr] = projection_from_entries(
            entries, local_dim=local_dim, global_dim=len(tables[attr])
        )

    return projectors
