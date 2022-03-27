"""Functions and factories to create symbolic variables."""

from sysopt.backends import sparse_matrix, SymbolicVector, is_symbolic, cast
from sysopt.block import Block


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

    def __hash__(self):
        return id(self)


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
