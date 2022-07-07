"""Module for converting sysopt expression graphs into casadi functions."""

from typing import Dict

import casadi
from sysopt.symbolic import (
    is_matrix, recursively_apply, SymbolicAtom, ExpressionGraph, GraphWrapper
)


def substitute(graph,
               symbols: Dict[SymbolicAtom, casadi.SX]):

    def leaf_function(obj):
        if obj in symbols:
            return symbols[obj]
        if is_matrix(obj) or isinstance(obj, (int, float, complex)):
            return casadi.SX(obj)
        raise NotImplementedError(f'Don\'y know how to evaluate {obj}')

    def trunk_function(op, *children):
        return op(*children)

    return recursively_apply(graph, trunk_function, leaf_function)
