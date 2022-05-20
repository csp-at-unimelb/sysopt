from typing import List, Union
from sysopt.symbolic import Algebraic, scalar_shape, is_op, is_matrix

import casadi as _casadi


def lambdify(graph,
             arguments: List[Union[Algebraic, List[Algebraic]]],
             name: str = 'f'):

    substitutions = {}
    for i, arg in enumerate(arguments):
        if isinstance(arg, list):
            assert all(sub_arg.shape == scalar_shape for sub_arg in arg), \
                'Invalid arguments, lists must be a list of scalars'
            symbol = _casadi.SX.sym(f'x_{i}', (len(arg), 1))
            substitutions.update(
                {sub_arg: symbol[j] for j, sub_arg in enumerate(arg)})
        else:
            try:
                n,  = arg.shape
            except ValueError as ex:
                n, m = arg.shape
                if m > 1:
                    raise ex
            symbol = _casadi.SX.sym(f'x_{i}', (n, 1))
            substitutions[arg] = symbol

    def recurse(node):
        obj = graph.nodes[node]
        if is_op(obj):
            args = [recurse(child) for child in graph.edges[node]]
            return obj(*args)
        if is_matrix(obj):
            return _casadi.SX(obj)
        try:
            return substitutions[obj]
        except (KeyError, TypeError):
            return obj

    expressions = recurse(graph.head)

    try:
        outputs = [_casadi.vertcat(expr) for expr in expressions]
    except Exception:
        outputs = [expressions]
    return _casadi.Function(name,  list(substitutions.values()), outputs)
