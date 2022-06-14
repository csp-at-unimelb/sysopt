"""Module for converting sysopt expression graphs into casadi functions."""

from typing import Dict

import casadi
from sysopt.symbolic import is_matrix, recursively_apply, SymbolicAtom


def substitute(graph,
               symbols: Dict[SymbolicAtom, casadi.SX]):

    def leaf_function(obj):
        if obj in symbols:
            return symbols[obj]
        if is_matrix(obj):
            return casadi.SX(obj)
        raise NotImplementedError(f'Don\'y know how to evaluate {obj}')

    def trunk_function(op, *children):
        return op(*children)


    expression = recursively_apply(graph, trunk_function, leaf_function)

    return expression
