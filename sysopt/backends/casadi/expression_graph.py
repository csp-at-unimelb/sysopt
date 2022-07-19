"""Module for converting sysopt expression graphs into casadi functions."""

from typing import Dict, List

import casadi
from sysopt.symbolic import is_matrix, recursively_apply, \
    SymbolicAtom, ExpressionGraph, Algebraic, GraphWrapper

from .function import compiles


def substitute(graph: ExpressionGraph,
               symbols: Dict[SymbolicAtom, casadi.SX]):

    def leaf_function(obj):
        if is_matrix(obj) or isinstance(obj, (int, float, complex)):
            return casadi.SX(obj)
        if obj in symbols:
            return symbols[obj]
        raise NotImplementedError(f'Don\'y know how to evaluate {obj}')

    def trunk_function(op, *children):
        return op(*children)

    return recursively_apply(graph, trunk_function, leaf_function)


@compiles(GraphWrapper)
def compile_expression_graph(obj: GraphWrapper):
    return CasadiFunctionWrapper(obj.graph, obj.arguments)


class CasadiFunctionWrapper(Algebraic):
    def __init__(self,
                 graph: ExpressionGraph,
                 arguments: List[SymbolicAtom],
                 name: str = 'f'):
        self._shape = graph.shape
        self._symbols = {
            a: casadi.SX.sym(str(a), *a.shape) for a in arguments
        }

        f_impl = substitute(graph, self._symbols)
        self.func = casadi.Function(name,
                                    list(self._symbols.values()),
                                    [f_impl])

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return repr(self.func)

    @property
    def shape(self):
        return self._shape

    def symbols(self):
        return set(self._symbols.keys())

    def __call__(self, *args):
        result = self.func(*args)
        try:
            return result.full()
        except AttributeError:
            return result

    def pushforward(self, *args):
        n = len(self.symbols())
        assert len(args) == 2 * n, f'expected {2 * n} arguments, ' \
                                   f'got {len(args)}'
        x, dx = args[:n], args[n:]
        out_sparsity = casadi.DM.ones(*self.shape)
        jac = self.func.jacobian()(*x, out_sparsity)
        dx = casadi.vertcat(*dx)
        result = jac @ dx

        return self.func(*x), result

