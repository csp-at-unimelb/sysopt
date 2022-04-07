"""Interface for Symbolic Functions and AutoDiff."""

import copy
from typing import Callable, Union, Iterable

from sysopt.types import Domain

from sysopt.block import Block, Composite
from sysopt.helpers import flatten, strip_nones

from sysopt.symbolic.symbols import as_vector
from sysopt.symbolic.function_ops import (
    FunctionOp, Concatenate, project, compose)

from sysopt.backends import (
    concatenate_symbols, SymbolicVector, is_symbolic
)


def coproduct(*functions: 'FunctionOp'):
    r"""Returns the functional coproduct of arguments

    For :math:`f_i:\oplus_j X^i_j ->Y_i`, this method
    returns a function :math:`F: \oplus_j (\oplus_iX^_j) -> \oplus_i Y_i`

    That is, the domain of the $j$th $X_j$ argument of $F$ is the direct sum
    (over $i$) of the $j$th arguments of $f_i$, $X_j = \oplus_i X^i_j$.

    When calling the returned function, the stacked input vectors are
    split into their target domains, then passed to the respective functions.

    Args:
        *functions: FunctionOps with the same number of arguments.

    Returns: Operation that computes the function coproduct.

    """
    klasses = {f.__class__ for f in functions}
    if all(issubclass(k, FunctionOp) for k in klasses):
        return BlockFunctionWrapper.coproduct(functions)
    if all(issubclass(k, VectorFunctionWrapper) for k in klasses):
        return VectorFunctionWrapper.coproduct(functions)

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
        return self.function(args)

    @staticmethod
    def coproduct(args: Iterable['VectorFunctionWrapper']):
        domain = 0
        codomain = 0
        func_list = []

        for func in args:
            func_list.append(
                (func, slice(domain, domain + func.domain),
                 slice(codomain, codomain + func.codomain))
            )
            domain += func.domain
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


def concatenate(*args):
    flat_args = strip_nones(flatten(args, 2))
    if any(isinstance(arg, FunctionOp) for arg in flat_args):
        return Concatenate(*flat_args)
    if any(is_symbolic(arg) for arg in flat_args):
        return concatenate_symbols(*flat_args)
    if all(isinstance(arg, (float, int)) for arg in flat_args):
        return flat_args

    raise NotImplementedError


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

    def __call__(self, *args):
        args = [
            as_vector(a) for a in args
        ]
        return self.function(*args)

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
    def coproduct(args: Iterable['BlockFunctionWrapper']):
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

            if any(is_symbolic(v) for v in args):
                result = SymbolicVector('r', codomain)
            else:
                result = [0] * codomain

            for sl, f in func_list:
                result[sl] = f(*args)

            return result

        return BlockFunctionWrapper(
            domain=d,
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
            domain=1,
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

    return x0, f, g, h


def create_functions_from_block(block: Union[Block, Composite]):
    try:
        functions = {component: create_functions_from_block(component)
                     for component in block.components}
    except AttributeError:
        # block is a leaf block
        return _create_functions_from_leaf_block(block)
    domain_offsets = {}
    output_offsets = {}
    output_codomain = 0
    domain = Domain()
    for component, (_, f, g, h) in functions.items():
        comp_domain = None
        for func in (f, g, h):
            if not func:
                continue
            comp_domain = func.domain
        if not comp_domain:
            raise TypeError(f'Block {component} nas no functions')
        domain_offsets[component] = copy.copy(domain)
        domain += comp_domain
        if g:
            output_offsets[component] = output_codomain
            output_codomain += g.codomain

    lists = zip(*list(functions.values()))
    x0_list, f_list, g_list, h_list = [strip_nones(l) for l in lists]
    h = coproduct(*h_list) if h_list else None
    x0 = coproduct(*x0_list) if x0_list else None
    f = coproduct(*f_list) if f_list else None
    g = coproduct(*g_list) if g_list else None

    if not block.wires:
        return x0, f, g, h

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
        (src, dest) for src, dest in block.wires
        if src.port_type != dest.port_type
    ]

    for src, dest in in_wires:
        component_offset = domain_offsets[dest.parent].inputs
        for input_index, component_index in zip(dest.indices, src.indices):
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

    for src, dest in internal_wires:
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

    f = compose(f, arg_permute)
    if h:
        h = concatenate(compose(h, arg_permute), *h_new)
    else:
        h = concatenate(*h_new)
    if g_actual:
        g = concatenate(*(g_actual[i] for i in range(len(block.outputs))))
    else:
        g = None

    return x0, f, g, h
