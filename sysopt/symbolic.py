"""Interface for Symbolic Functions and AutoDiff."""

# pylint: disable=wildcard-import,unused-wildcard-import
import dataclasses

from sysopt.backends import *
from sysopt.block import Block, Composite
from typing import Tuple, Callable, Union, Iterable


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


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
        return arg,


@dataclasses.dataclass
class Domain:
    time: int
    states: int
    constraints: int
    inputs: int
    parameters: int

    def __iter__(self):
        return iter((self.time, self.states, self.constraints,
                    self.inputs, self.parameters))

    def __getitem__(self, item):

        return list(self)[item]

    def __iadd__(self, other):
        self.states += other.states
        self.constraints += other.constraints
        self.inputs += other.inputs
        self.parameters += other.parameters
        return self

    def __add__(self, other):
        obj = Domain(*self)
        obj += other
        return obj

    def __eq__(self, other):
        return all(i == j for i, j in zip(self, other))


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
    def concatenate(args: Iterable['SimpleFunctionWrapper']):
        domain = 0
        codomain = 0
        func_list = []
        for func in args:

            p = lambda x: func(x[domain: domain + func.domain]), \
                slice(codomain, codomain+ func.codomain)
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


class BlockFunctionWrapper:
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
        self.domain = domain
        self.codomain = codomain

        self.function = function

    def __call__(self, *args):
        args = [
            as_vector(a) for a in args
        ]
        return self.function(args)

    @staticmethod
    def concatenate(args: Iterable['BlockFunctionWrapper']):
        d = Domain(1, 0, 0, 0, 0)

        func_list = []
        codomain = 0
        for func in args:
            _, n_x, n_z, n_u, n_p = func.domain
            func_list.append(
                lambda *X: func(X[0],
                                X[1][d.states:d.states + n_x],
                                X[2][d.constraints: d.constraints + n_z],
                                X[3][d.inputs: d.inputs + n_u],
                                X[4][d.parameters: d.parameters + n_p])
            )
            d += func.domain
            codomain += func.codomain

        def call(*args):
            return concatenate(*[func(*args) for func in func_list])

        return BlockFunctionWrapper(
            domain=d,
            codomain=codomain,
            function=call
        )


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
            function=lambda args: block.initial_state(args[-1])
        )
        f = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.state,
            function=lambda args: block.compute_dynamics(*args)
        )

    if block.signature.outputs > 0:
        g = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.outputs,
            function=lambda args: block.compute_outputs(*args)
        )
    if block.signature.constraints > 0:
        h = BlockFunctionWrapper(
            domain=domain,
            codomain=block.signature.constraints,
            function=lambda args: block.compute_residuals(*args)
        )

    return x0, f, g, h


def create_functions_from_block(block: Union[Block, Composite]):
    try:
        functions = [create_functions_from_block(component)
                     for component in block.components]
    except AttributeError:
        # block is a leaf block
        return _create_functions_from_leaf_block(block)

    x0_list, f_list, g_list, h_list = zip(*functions)
    x0, f, g, h = None, None, None, None
    if x0_list:
        x0 = SimpleFunctionWrapper.concatenate(x0_list)

    if f_list:
        f = BlockFunctionWrapper.concatenate(f_list)

    if g_list:
        g = BlockFunctionWrapper.concatenate(g_list)

    if h_list:
        h = BlockFunctionWrapper.concatenate(h_list)

    return x0, f, g, h
