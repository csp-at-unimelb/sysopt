"""Interface for Symbolic Functions and AutoDiff."""

# pylint: disable=wildcard-import,unused-wildcard-import
from sysopt.backends import *
from sysopt.block import Block
from typing import NamedTuple, Tuple, Callable, Union


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


Domain = NamedTuple(
    'Domain',
    [('time', int),
     ('states', int),
     ('constraints', int),
     ('inputs', int),
     ('parameters', int)]
)


class FunctionWrapper:
    """Wrapper for differentiable functions.

    Args:
        domain:     The dimensions of the input space.
        codomain:   The dimensions of the output space.
        function:   The function definition.

    """
    def __init__(self,
                 domain: Union[Domain, int],
                 codomain: Tuple[int, ...],
                 function: Callable):
        self.domain = domain
        self.codomain = codomain

        self.function = function

    def __call__(self, *args):
        args = [
            as_vector(a) for a in args
        ]
        return self.function(args)


def create_functions_from_block(block: Block):
    if hasattr(Block, 'components'):
        raise NotImplementedError

    domain = (
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
        x0 = FunctionWrapper(
            domain=1,
            codomain=(block.signature.state, block.signature.constraints),
            function=lambda args: block.initial_state(args[-1])
        )
        f = FunctionWrapper(
            domain=domain,
            codomain=block.signature.state,
            function=lambda args: block.compute_dynamics(*args)
        )

    if block.signature.outputs > 0:
        g = FunctionWrapper(
            domain=domain,
            codomain=block.signature.outputs,
            function=lambda args: block.compute_outputs(*args)
        )
    if block.signature.constraints > 0:
        h = FunctionWrapper(
            domain=domain,
            codomain=block.signature.constraints,
            function=lambda args: block.compute_residuals(*args)
        )

    return x0, f, g, h
