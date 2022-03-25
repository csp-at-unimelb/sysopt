"""Interface for Symbolic Functions and AutoDiff."""

# pylint: disable=wildcard-import,unused-wildcard-import

from abc import ABCMeta
import copy
from typing import Tuple, Callable, Union, Iterable

from sysopt.types import Domain
from sysopt.backends import *
from sysopt.block import Block, Composite
from sysopt.helpers import flatten, strip_nones


def require_equal_domains(func):
    def validator(a, b):
        if a.domain != b.domain:
            msg = f'Domains {a.domain} != {b.domain} for arguments {a}, {b}'
            raise TypeError(msg)
        return func(a, b)
    return validator


def require_equal_order_codomain(func):
    def validator(a, b):
        if ((isinstance(a.codomain, int) and isinstance(b.codomain, int))
                or len(a.codomain) == len(b.codomain)):
            return func(a, b)
        else:
            msg = f'Codomains {a.codomain}, {b.codomain} are not compatible'\
                  f'for arguments {a}, {b}'
            raise TypeError(msg)

    return validator


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


def coproduct(*functions):
    klasses = {f.__class__ for f in functions}
    if all(issubclass(k, TensorOp) for k in klasses):
        return BlockFunctionWrapper.coproduct(functions)
    if all(issubclass(k, SimpleFunctionWrapper) for k in klasses):
        return SimpleFunctionWrapper.coproduct(functions)

    raise NotImplementedError


class DecisionVariable(SymbolicVector):
    """Symbolic variable for specifying optimisation targets.

    Decision variables are either free, or bound to a block and parameter.
    Free decision variables can be created using::

        variable = DecisionVariable()

    Decision variables that are bound to a model parameter can be created via::

        bound_var = DecisionVariable(block, param_index_or_name)

    Where `param_index_or_name` is the `int` index, or `str` name of the
    parameter to be used in optimisation.


    Args:
        args: Optional Block or tuple of Block and parameter

    """
    _counter = 0
    is_symbolic = True

    def __new__(cls, *args):
        name = f'w{DecisionVariable._counter}'
        DecisionVariable._counter += 1
        is_free_variable = len(args) == 0
        is_block_params = not is_free_variable and isinstance(args[0], Block)
        is_block_vector = is_block_params and len(args) == 1
        is_block_single = is_block_params and isinstance(args[1], (int, str))
        is_valid = (is_block_vector or is_block_single or is_free_variable)

        assert is_valid, 'Invalid parameter definition'

        if is_block_vector:
            obj = SymbolicVector.__new__(
                cls, name, args[0].signature.parameters)
            setattr(
                obj, 'parameter', (args[0],
                                   slice(0, args[0].signature.parameters))
            )

        elif is_block_single:
            obj = SymbolicVector.__new__(cls, name, 1)
            block, param = args
            if isinstance(param, str):
                idx = block.metadata.parameters.index(param)
                if idx < 0:
                    raise ValueError(
                        f'Invalid parameter for {block}: {param} not found'
                    )
            elif isinstance(param, int):
                idx = param
            else:
                raise ValueError(
                    f'Invalid parameter for {block}: {param} not found'
                )
            setattr(obj, 'parameter', (block, slice(idx, idx + 1)))
        else:
            obj = SymbolicVector.__new__(cls, name, 1)

        return obj


def as_vector(arg):
    try:
        len(arg)
        return arg
    except TypeError:
        if isinstance(arg, (int, float)):
            return arg,
    if is_symbolic(arg):
        return cast(arg)

    raise NotImplementedError(
        f'Don\'t know to to vectorise {arg.__class__}'
    )


class SimpleFunctionWrapper:
    def __init__(self,
                 domain: int,
                 codomain: Union[int, Tuple[int, int], Tuple[int, ...]],
                 function: Callable
                 ):
        self.domain = domain
        self.codomain = codomain
        self.function = function

    def __call__(self, *args):
        return self.function(args)

    @staticmethod
    def coproduct(args: Iterable['SimpleFunctionWrapper']):
        domain = 0
        codomain = 0
        func_list = []
        for func in args:
            p = lambda x: func(x[domain: domain + func.domain]), \
                slice(codomain, codomain + func.codomain)
            domain += func.domain
            codomain += func.codomain
            func_list.append(p)

        def call(x):
            result = [0] * codomain
            for f, slc in func_list:
                result[slc] = f(x)
            return result

        return SimpleFunctionWrapper(
            domain,
            codomain,
            call
        )


def concatenate(*args):
    flat_args = strip_nones(flatten(args, 2))
    if any(isinstance(arg, TensorOp) for arg in flat_args):
        return Concatenate(*flat_args)
    if any(is_symbolic(arg) for arg in flat_args):
        return concatenate_symbols(*flat_args)
    if all(isinstance(arg, (float, int)) for arg in flat_args):
        return flat_args

    raise NotImplementedError


class TensorOp(metaclass=ABCMeta):
    @require_equal_domains
    def __sub__(self, other):
        return Subtract(self, other)


class Concatenate(TensorOp):
    def __init__(self, *args: TensorOp):
        op, *remainder = args
        self.__vectors = [op]
        self.domain = op.domain.copy()
        self.codomain = op.codomain
        for arg in args[1:]:
            self.append(arg)

    @require_equal_domains
    @require_equal_order_codomain
    def append(self, other):
        self.__vectors.append(other)
        self.codomain += other.codomain

    def __call__(self, *args):
        result = [f(*args) for f in self.__vectors]
        return concatenate(result)


class BlockFunctionWrapper(TensorOp):
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


class ArgPermute(TensorOp):
    def __init__(self, domain):
        self.codomain = Domain(*domain)
        self.domain = Domain(*domain)
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
        block.signature.state,
        block.signature.constraints,
        block.signature.inputs,
        block.signature.parameters
    )
    x0 = None
    f = None
    g = None
    h = None
    if block.signature.state > 0:
        x0 = SimpleFunctionWrapper(
            domain=1,
            codomain=block.signature.state,
            function=block.initial_state
        )
        f = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.state,
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


class TensorProjection(TensorOp):
    def __init__(self,
                 domain: Domain,
                 variable: int,
                 index: int):
        self.codomain = 1
        self.domain = domain
        self.index = index
        self.variable = variable

    def __call__(self, *args):
        return args[self.variable][self.index]


class VectorProjection(TensorOp):
    def __init__(self, domain, index):
        self.codomain = 1
        self.domain = domain
        self.index = index

    def __call__(self, *vector):
        return vector[self.index]


def project(domain, *args):
    if isinstance(domain, int):
        index, = args
        return VectorProjection(domain, index)
    else:
        variable, index = args
        space = Domain.index_of_field(variable)
        assert space >= 0
        return TensorProjection(domain, space, index)


class Subtract(TensorOp):
    def __init__(self, lhs, rhs):
        assert lhs.domain == rhs.domain
        assert lhs.codomain == rhs.codomain
        self.lhs = lhs
        self.rhs = rhs
        self.domain = copy.copy(lhs.domain)
        self.codomain = copy.copy(lhs.codomain)

    def __call__(self, *args):
        return self.lhs(*args) - self.rhs(*args)


def subtract(lhs, rhs):
    return Subtract(lhs, rhs)


class Compose(TensorOp):
    def __init__(self, outer, inner):

        if inner.codomain != outer.domain:
            msg = f'Cannot compose {outer} with {inner}. '\
                  f'Inner has codomain {inner.codomain}, '\
                  f'while outer has domain {outer.domain}'
            raise TypeError(msg)

        self.inner = inner
        self.outer = outer

    @property
    def domain(self):
        return self.inner.domain

    @property
    def codomain(self):
        return self.outer.codomain

    def __call__(self, *args):
        r = self.inner(*args)
        try:
            return self.outer(*r)
        except TypeError as ex:
            print(f'Error calling {self.outer} with {r}')

            raise ex


def compose(*args):
    outer, inner, *remainder = args
    if not remainder:
        return Compose(outer, inner)
    else:
        return compose(Compose(outer, inner), *remainder)


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
