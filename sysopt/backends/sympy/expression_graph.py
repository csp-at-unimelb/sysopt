"""Module for converting sysopt expression graphs into casadi functions."""

from typing import Dict, List

import sympy
from sysopt.symbolic import (
    is_matrix, recursively_apply, Variable, ExpressionGraph, Algebraic,
    GraphWrapper, Function, Composition, ConstantFunction)

from sysopt.backends.impl_hooks import implements, get_implementation


def substitute(graph: ExpressionGraph,
               symbols: Dict[Variable, sympy.Symbol]):

    def leaf_function(obj):
        if is_matrix(obj):
            return sympy.ImmutableSparseMatrix(*obj.shape, obj)

        if isinstance(obj, (int, float, complex)):
            return obj

        if obj in symbols:
            return symbols[obj]

        if isinstance(obj, (Function, Composition)):
            arguments = {a: symbols[a] for a in obj.arguments}
            impl = get_implementation(obj)
            return impl.call(arguments)

        raise NotImplementedError(f'Don\'y know how to evaluate {obj} of'
                                  f'type {type(obj)}')

    def trunk_function(op, *children):
        r = op(*children)
        try:
            if r.shape == (1, 1):
                r = r[0, 0]
            elif r.shape == (1,):
                r = r[0]
        except AttributeError:
            pass

        return r

    return recursively_apply(graph, trunk_function, leaf_function)


@implements(ConstantFunction)
def to_constant(func: ConstantFunction):

    if is_matrix(func.value):
        v = sympy.ImmutableSparseMatrix(func.value)
    else:
        v = float(func.value)
    return lambda x: v


@implements(ExpressionGraph)
def to_sympy_eqn(graph: ExpressionGraph):
    symbols = {
        s: sympy.Matrix(
            [sympy.symbols(f'{s.name}_{i}') for i in range(s.shape[0])]
        )
        for s in graph.symbols()
    }

    return substitute(graph, symbols)


# @implements(GraphWrapper)
# def compile_expression_graph(obj: GraphWrapper):
#     return CasadiGraphWrapper(obj.graph, obj.arguments)


# def to_function(obj: GraphWrapper, name='f'):
#     symbols = {
#         s: casadi.MX.sym(str(s), *s.shape) for s in obj.arguments
#     }
#     impl = substitute(obj.graph, symbols)
#     return casadi.Function(
#         name,
#         list(symbols.values()),
#         [impl]
#     )


# class CasadiGraphWrapper(Algebraic):
#     """Casadi function wrapper for a function compled from an
#     expression graph."""
#
#     def __init__(self,
#                  graph: ExpressionGraph,
#                  arguments: List[Variable],
#                  name: str = 'f'):
#         self._shape = graph.shape
#         self._symbols = {
#             a: casadi.MX.sym(str(a), *a.shape) for a in arguments
#         }
#
#         f_impl = substitute(graph, self._symbols)
#         self.func = casadi.Function(name,
#                                     list(self._symbols.values()),
#                                     [f_impl])
#
#     def __hash__(self):
#         return id(self)
#
#     def __repr__(self):
#         return repr(self.func)
#
#     @property
#     def shape(self):
#         return self._shape
#
#     def symbols(self):
#         return set(self._symbols.keys())
#
#     def __call__(self, *args):
#         result = self.func(*args)
#         try:
#             return result.full()
#         except AttributeError:
#             return result
#
#     def pushforward(self, *args):
#         n = len(self.symbols())
#         assert len(args) == 2 * n, f'expected {2 * n} arguments, ' \
#                                    f'got {len(args)}'
#         x, dx = args[:n], args[n:]
#         out_sparsity = casadi.DM.ones(*self.shape)
#         jac = self.func.jacobian()(*x, out_sparsity)
#         dx = casadi.vertcat(*dx)
#         result = jac @ dx
#
#         return self.func(*x), result
#
