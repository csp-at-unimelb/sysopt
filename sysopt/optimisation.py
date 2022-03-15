from typing import Callable, Optional
from sysopt.types import Numeric
from sysopt import symbolic
from sysopt.block import Block


class DecisionVariable:
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
            obj = symbolic.symbol(name, args[0].signature.parameters)
            setattr(obj,
                    'parameter',
                    (args[0], slice(0, args[0].signature.parameters))
            )
        elif is_block_single:
            obj = symbolic.symbol(name, 1)
            block, param = args
            if isinstance(param, str):
                idx = block.metadata.parameters.index(param)
                if idx < 0:
                    raise ValueError(
                        f"Invalid parameter for {block}: {param} not found"
                    )
            elif isinstance(param, int):
                idx = param
            else:
                raise ValueError(
                    f"Invalid parameter for {block}: {param} not found"
                )
            setattr(obj, 'parameter',(block, slice(idx, idx + 1)))
        else:
            obj = symbolic.symbol(name, 1)

        return obj


class Minimise:
    def __init__(self, cost, subject_to=None):
        self.cost = cost
        self.constraints = subject_to or []

    @property
    def decision_variables(self):
        atoms = set(symbolic.list_symbols(self.cost))
        for constraint in self.constraints:
            atoms |= set(symbolic.list_symbols(constraint))

        return atoms
