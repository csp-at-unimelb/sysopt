"""Symbolic Variables and Functions for optmisation problems."""

import weakref
from typing import Union, Tuple, Optional
from sysopt.symbolic.core import (
    Variable, scalar_shape, ExpressionGraph, concatenate, Quadrature,
    Algebraic
)


def find_param_index_by_name(block, name: str):
    try:
        return block.find_by_name('parameters', name)
    except (AttributeError, ValueError):
        pass
    try:
        return block.parameters.index(name)
    except ValueError:
        pass
    try:
        return block.parameters.index(f'{str(block)}/{name}')
    except ValueError:
        pass
    raise ValueError(f'Could not find parameter \'{name}\' in block {block}.')


class PiecewiseConstantSignal(Variable):
    """
    Args:
        name:
        frequency: Update rate (in hertz)
        shape: Vector dimensions of this variable (must be of the form `(d,)`
            where `d` is the dimension.

    """
    def __init__(self, name=None, frequency=1, shape=scalar_shape):
        super().__init__(name, shape)
        self.frequency = frequency


def symbolic_vector(name, length=1):
    return Variable(name, shape=(length, ))


def resolve_parameter_uid(block, index):
    name = block.parameters[index]
    return hash(name)


class Parameter(Variable):
    """Symbolic type for variables bound to a block parameter.

    Args:
        block: The model block from which to derive the symbolic parameter.
        parameter: Index or name of the desired symbolic parameter.

    """
    _table = {}

    def __new__(cls, block, parameter: Union[str, int]):

        if isinstance(parameter, str):
            index = find_param_index_by_name(block, parameter)
        else:
            index = parameter
        assert 0 <= index < len(block.parameters),\
            f'Invalid parameter index for {block}: got {parameter},'\
            f'expected a number between 0 and {len(block.parameters)}'

        uid = resolve_parameter_uid(block, index)

        try:
            obj = Parameter._table[uid]
            return obj
        except KeyError:
            pass
        assert isinstance(index, int)
        obj = Variable.__new__(cls)
        setattr(obj, 'uid', uid)
        setattr(obj, 'index', index)
        setattr(obj, '_parent', weakref.ref(block))
        Parameter._table[uid] = obj
        obj.__init__(name=None)
        return obj

    def __hash__(self):
        return hash(self.uid)

    def __cmp__(self, other):
        try:
            return self.uid == other.uid
        except AttributeError:
            return False

    def get_source_and_slice(self):
        return self._parent(), slice(self.index, self.index + 1, None)

    @property
    def name(self):
        parent = self._parent()
        return parent.parameters[self.index]

    def __repr__(self):
        return self.name

    @property
    def shape(self):
        return scalar_shape

    def symbols(self):
        return {self}

    @staticmethod
    def from_block(block):
        return [Parameter(block, i) for i in range(len(block.parameters))]


def extract_quadratures(graph: Union[ExpressionGraph, Quadrature]) \
        -> Tuple[Algebraic, Optional[Variable], Optional[ExpressionGraph]]:
    r"""Split an expression graph into algebraic and integral terms.

    Args:
        graph - The expression graph `g` that may contain integrals

    Returns:
         a (possible new) graph, quadrature variables, and integrands


    As an example, the function :math:`f(t, x) = x(t) +\int_0^t t(\tau)d\tau`
    will be split into

    math::
        f(t, x) = x(t) + q(t)\\
        \dot{q} = x(t)

    The 'new' graph will be `f(t, x, q) = x(q) + q(t)`,
    the quadrature variable will be `q`, and the integrand will be `x(t)`.

    """
    quadratures = {}
    if isinstance(graph, Variable):
        return graph, None, None

    if isinstance(graph, Quadrature):
        q = Variable('q', shape=graph.integrand.shape)
        return q, q, graph.integrand

    def recurse(node_idx):
        node = graph.nodes[node_idx]
        if isinstance(node, Quadrature):
            q = Variable('q', shape=node.integrand.shape)
            quadratures[q] = node.integrand
            return q
        elif node_idx not in graph.edges:
            return node
        else:
            return ExpressionGraph(
                graph.nodes[node_idx],
                *[recurse(idx) for idx in graph.edges[node_idx]]
            )

    out_graph = recurse(graph.head)
    if len(quadratures) == 0:
        return graph, None, None

    if len(quadratures) == 1:
        q, integrand = list(quadratures.items())[0]
        return out_graph, q, integrand

    dot_q = concatenate(*quadratures.values())
    vector_q = Variable('q', shape=dot_q.shape)
    offset = 0
    for q in quadratures:
        try:
            n, = q.shape
        except ValueError:
            n, m = q.shape
            assert m == 1

        quadratures[q] = vector_q[offset: offset + n]
        offset += n
    out_graph = out_graph.call(quadratures)
    return out_graph, vector_q, dot_q


