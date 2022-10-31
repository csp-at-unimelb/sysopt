"""Module for converting sysopt expression graphs into casadi functions."""

from typing import Dict, List

import sympy
from sysopt.symbolic import (
    is_matrix, recursively_apply, Variable, ExpressionGraph, Algebraic,
    GraphWrapper, Function, Composition, ConstantFunction, matmul)

from sysopt.backends.implementation_hooks import implements, get_implementation
from sysopt.backends.sympy.helpers import sympy_vector


def float_to_int(eq):
    # pylint: disable=line-too-long
    """Convert floats that are ints to ints

    see: https://stackoverflow.com/questions/64761602/how-to-get-sympy-to-replace-ints-like-1-0-with-1
    """
    reps = {}
    e = eq.replace(
        lambda x: x.is_Float and x == int(x),
        lambda x: reps.setdefault(x, sympy.Dummy()))
    return e.xreplace({v: int(k) for k, v in reps.items()})


def to_scalar(obj):
    if int(obj) == obj:
        return float_to_int(sympy.Number(int(obj)))
    else:
        return sympy.Number(obj)


def substitute(graph: ExpressionGraph,
               symbols: Dict[Variable, sympy.Symbol]):

    def leaf_to_sympy_obj(obj):
        if is_matrix(obj):
            if obj.shape in ((1, ), (1, 1)):
                # hack: for some reason sympy doesn't like to (1, ) matricies.
                return to_scalar(obj.ravel()[0])
            try:
                return float_to_int(
                    sympy.ImmutableSparseMatrix(*obj.shape, obj)
                )
            except TypeError as ex:
                raise TypeError(
                    f'Failed to convert {obj} to a sympy matrix') from ex

        if isinstance(obj, (int, float, complex)):
            return to_scalar(obj)

        try:
            return symbols[obj]
        except KeyError:
            pass

        if isinstance(obj, (Function, Composition)):
            arguments = {a: symbols[a] for a in obj.arguments}
            impl = get_implementation(obj)
            return impl.call(arguments)

        raise NotImplementedError(f'Don\'y know how to evaluate {obj} of'
                                  f'type {type(obj)}')

    def trunk_to_sympy_obj(op, *children):
        # Hack to get around sympy.Mul a not having @ defined.
        if op is matmul:
            r = children[0]
            for child in children[1:]:
                try:
                    r = r @ child
                except TypeError:
                    r = r * child
        else:
            r = op(*children)

        try:
            if r.shape == (1, 1):
                r = r[0, 0]
            elif r.shape == (1,):
                r = r[0]
        except AttributeError:
            pass

        return r

    return recursively_apply(graph, trunk_to_sympy_obj, leaf_to_sympy_obj)


def expand_substitutions(matrix_symbol, matrix_values):
    try:
        assert matrix_symbol.shape == matrix_values.shape
    except AttributeError:
        if matrix_values is None:
            return []
        else:
            return [(matrix_symbol, matrix_values)]

    return [(matrix_symbol[i, j], matrix_values[i, j])
            for i in range(matrix_symbol.shape[0])
            for j in range(matrix_symbol.shape[1])]


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
        s: sympy_vector(s.name, s.shape) for s in graph.symbols()
    }

    return substitute(graph, symbols)


@implements(GraphWrapper)
def compile_expression_graph(obj: GraphWrapper):
    return SympyGraphWrapper(obj.graph, obj.arguments)


class SympyGraphWrapper(Algebraic):
    """Function wrapper for a function as an expression graph."""

    def __init__(self,
                 graph: ExpressionGraph,
                 arguments: List[Variable],
                 name: str = 'f'):
        self._shape = graph.shape
        self._symbols = {
            a: sympy_vector(a.name, a.shape) for a in arguments
        }
        self.name = name
        self.func = substitute(graph, self._symbols)

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
        assert len(args) == len(self._symbols)

        subs = []
        for (atom, value) in zip(self._symbols.values(), args):
            subs += expand_substitutions(atom, value)

        f = self.func.subs(subs)

        return f

    def pushforward(self, *args):
        n = len(self.symbols())
        assert len(args) == 2 * n, f'expected {2 * n} arguments, ' \
                                   f'got {len(args)}'
        x, dx = args[:n], args[n:]
        jac = sympy.Matrix(
            [sympy.diff(self.func, a) for a in self._symbols.values()]
        )

        result = jac.evalf(x) @ dx

        return self.func.evalf(x), result
