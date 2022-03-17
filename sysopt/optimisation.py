"""Classes and functions for system optimisation."""

from sysopt import symbolic
from sysopt.block import Block


class DecisionVariable:
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
            obj = symbolic.SymbolicVector(name, args[0].signature.parameters)
            setattr(
                obj, 'parameter', (args[0],
                                   slice(0, args[0].signature.parameters))
            )

        elif is_block_single:
            obj = symbolic.SymbolicVector(name, 1)
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
            obj = symbolic.SymbolicVector(name, 1)

        return obj


class Minimise:
    """Problem statement for single objective constrained optimisation.

    Args:
        cost: Symbolic expression for the cost function.

    Keyword Args:
        subject_to: Option list of symbolic inequalities.

    """
    def __init__(self, cost, subject_to=None):
        self.cost = cost
        self.constraints = subject_to or []

    @property
    def decision_variables(self):
        atoms = set(symbolic.list_symbols(self.cost))
        for constraint in self.constraints:
            atoms |= set(symbolic.list_symbols(constraint))

        return atoms
