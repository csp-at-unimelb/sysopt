"""Functions and factories to create symbolic variables."""
import weakref
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from inspect import signature

import numpy as np
from scipy.sparse import dok_matrix, spmatrix
from typing import Union, List, Callable, Tuple, Optional

import sysopt.backends as backend
from sysopt.backends import SymbolicVector

epsilon = 1e-12


def find_param_index_by_name(block, name: str):
    try:
        return block.find_by_name('parameters', name)
    except ValueError:
        pass
    try:
        return block.parameters.index(name)
    except ValueError:
        pass
    raise ValueError(f'Could not find parameter {name} in block {block}.')


def sparse_matrix(shape: Tuple[int, int]):
    return dok_matrix(shape, dtype=float)


def is_symbolic(obj):
    try:
        return obj.is_symbolic
    except AttributeError:
        return backend.is_symbolic(obj)


def list_symbols(obj):
    try:
        return obj.symbols()
    except AttributeError:
        return backend.list_symbols(obj)


def projection_matrix(indices: List[int], dimension: int):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in enumerate(indices):
        matrix[i, j] = 1

    return matrix


__ops = defaultdict(list)
__shape_ops = {}

scalar_shape = (1, )


def infer_scalar_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    this_shape = shapes[0]
    for shape in shapes[1:]:
        if shape in (this_shape, scalar_shape):
            continue
        if this_shape == (1, ):
            this_shape = shape
        else:
            raise AttributeError('Invalid Shape')
    return this_shape


def matmul_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    n, m = shapes[0]
    for shape in shapes[1:]:
        try:
            n_next, m_next = shape
        except ValueError:
            n_next, = shape
            m_next = None
        if m != n_next:
            raise AttributeError('Invalid shape')
        else:
            m = m_next

    if m is not None:
        return n, m
    else:
        return n,


def transpose_shape(shape: Tuple[int, int]) -> Tuple[int, ...]:
    try:
        n, m = shape
    except ValueError:
        n, = shape
        m = 1
    return m, n


def infer_shape(op: Callable, *shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Infers the output shape from the operation on the given inputs."""
    return __shape_ops[op](*shapes)


def register_op(shape_func=infer_scalar_shape):
    """Decorator which register the operator as an expression graph op."""
    def wrapper(func):
        sig = signature(func)
        is_variable = any(
            param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            for param in sig.parameters.values())

        idx = None if is_variable else len(sig.parameters)
        __ops[idx].append(func)
        __shape_ops[func] = shape_func
        return func

    return wrapper


def wrap_as_op(func: Callable,
               arguments: Optional[int] = None,
               shape_func=infer_scalar_shape) -> Callable:
    """Wraps the function for use in expression graphs.

    Args:
        func:       A function to wrap
        arguments:  The number of arguments
        shape_func: A function which generates the output shape from the
            arguments.

    Returns:
        An callable operator for use in an expression graph.

    """
    __ops[arguments].append(func)

    def wrapper(*args):
        return ExpressionGraph(func, *args)

    __shape_ops[func] = shape_func

    return wrapper


@register_op()
def power(base, exponent):
    return base ** exponent


@register_op()
def add(lhs, rhs):
    return lhs + rhs


@register_op()
def sub(lhs, rhs):
    return lhs - rhs


@register_op(matmul_shape)
def matmul(lhs, rhs):
    return lhs @ rhs


@register_op()
def neg(obj):
    return -obj


@register_op()
def mul(lhs, rhs):
    return lhs * rhs


@register_op()
def div(lhs, rhs):
    return lhs / rhs


@register_op(transpose_shape)
def transpose(matrix):
    return matrix.T


def slice_to_list(slce: slice):
    return list(range(slce.stop))[slce]


class Inequality:
    """Inequality expression.

    Non-negative evaluation means that the inequality is satisfied.

    """
    def __init__(self, smaller, bigger):
        self.smaller = smaller
        self.bigger = bigger

    def __str__(self):
        return f'{self.smaller} <= {self.bigger}'

    def symbols(self):
        result = set()
        for term in (self.smaller, self.bigger):
            try:
                result |= term.symbols()
            except AttributeError:
                pass

        return result

    def to_graph(self):
        return self.bigger - self.smaller

    def call(self, args):
        return self.to_graph().call(args)


class PathInequality(Inequality):
    pass


class Algebraic(metaclass=ABCMeta):
    """Base class for symbolic terms in expression graphs."""
    def __init_subclass__(cls, **kwargs):
        setattr(cls, '__array_ufunc__', None)

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    def T(self):
        return ExpressionGraph(transpose, self)

    @abstractmethod
    def symbols(self):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        n = self.shape[0]
        if isinstance(item, slice):
            indices = slice_to_list(item)
        else:
            indices = [item]
        pi = projection_matrix(indices, n)
        return ExpressionGraph(matmul, pi, self)

    def __add__(self, other):
        return ExpressionGraph(add, self, other)

    def __radd__(self, other):
        return ExpressionGraph(add, other, self)

    def __neg__(self):
        return ExpressionGraph(neg, self)

    def __sub__(self, other):
        return ExpressionGraph(sub, self, other)

    def __rsub__(self, other):
        return ExpressionGraph(sub, other, self)

    def __matmul__(self, other):
        return ExpressionGraph(matmul, self, other)

    def __rmatmul__(self, other):
        return ExpressionGraph(matmul, other, self)

    def __mul__(self, other):
        return ExpressionGraph(mul, self, other)

    def __rmul__(self, other):
        return ExpressionGraph(mul, other, self)

    def __truediv__(self, other):
        return ExpressionGraph(div, self, other)

    def __rtruediv__(self, other):
        return ExpressionGraph(div, other, self)

    def __le__(self, other):
        return _less_or_equal(self, other)

    def __ge__(self, other):
        return _less_or_equal(other, self)

    def __gt__(self, other):
        return _less_or_equal(other, self + epsilon)

    def __lt__(self, other):
        return _less_or_equal(self, other + epsilon)

    def __cmp__(self, other):
        return id(self) == id(other)

    def __pow__(self, exponent, modulo=None):
        return ExpressionGraph(power, self, exponent)


def _less_or_equal(smaller, bigger):
    if is_temporal(smaller) or is_temporal(bigger):
        return PathInequality(smaller, bigger)
    else:
        return Inequality(smaller, bigger)


def is_op(value):
    try:
        return any(value in ops for ops in __ops.values())
    except ValueError:
        return False


class ExpressionGraph(Algebraic):
    """Graph representation of a symbolic expression."""

    def __init__(self, op, *args):
        self.nodes = []
        op_node = self.add_or_get_node(op)
        self.edges = {}
        self.edges.update(
            {op_node: [self.add_or_get_node(a) for a in args]}
        )
        self.head = op_node

    @property
    def shape(self):
        return self._get_shape_of(self.head)

    def _get_shape_of(self, node):
        if node in self.edges:
            op = self.nodes[node]
            shapes = [
                self._get_shape_of(child)
                for child in self.edges[node]
            ]
            return infer_shape(op, *shapes)
        obj = self.nodes[node]
        try:
            return obj.shape
        except AttributeError:
            if isinstance(obj, (float, int, complex)):
                return scalar_shape
        raise NotImplementedError(
            f'Don\'t know how to get the shape of {obj}'
        )

    def call(self, values):

        def recurse(node):
            obj = self.nodes[node]
            if is_op(obj):
                args = [recurse(child) for child in self.edges[node]]
                return obj(*args)
            try:
                return obj.call(values)
            except (AttributeError, TypeError):
                pass
            try:
                return values[obj]
            except (KeyError, TypeError):
                pass
            return obj

        return recurse(self.head)

    @property
    def is_symbolic(self):
        return self.symbols() != {}

    def add_or_get_node(self, value):
        if value is self:
            assert self.head is not None
            return self.head
        if is_op(value):
            idx = len(self.nodes)
            self.nodes.append(value)
            return idx
        if isinstance(value, ExpressionGraph):
            return self.merge_and_return_subgraph_head(value)
        try:
            return self.nodes.index(value)
        except ValueError:
            pass
        # else

        idx = len(self.nodes)
        self.nodes.append(value)
        return idx

    def merge_and_return_subgraph_head(self, other):
        new_indices = {
            old_idx: self.add_or_get_node(node)
            for old_idx, node in enumerate(other.nodes)
        }

        self.edges.update({
            new_indices[parent]: [new_indices[child] for child in children]
            for parent, children in other.edges.items()
        })
        return new_indices[other.head]

    def push_op(self, op, *nodes):
        op_node = self.add_or_get_node(op)
        node_indices = [self.add_or_get_node(node) for node in nodes]

        self.edges[op_node] = node_indices
        self.head = op_node
        return self

    def __iadd__(self, other):
        return self.push_op(add, self, other)

    def __isub__(self, other):
        return self.push_op(sub, self, other)

    def __imul__(self, other):
        return self.push_op(mul, self, other)

    def __idiv__(self, other):
        return self.push_op(div, self, other)

    def __imatmul__(self, other):
        return self.push_op(matmul, self, other)

    def __hash__(self):
        return hash((self.edges, *[hash(n) for n in self.nodes]))

    def symbols(self):

        def recurse(node):
            obj = self.nodes[node]
            if not is_op(obj):
                try:
                    return obj.symbols()
                except AttributeError:
                    return set()
            child_symbols = set.union(
                    *(recurse(child)
                      for child in self.edges[node])
                )
            if obj is evaluate_signal:
                return child_symbols - {get_time_variable()}
            else:
                return child_symbols

        return recurse(self.head)


class Variable(Algebraic):
    """Symbolic type for a free variable."""
    is_symbolic = True
    __array_ufunc__ = None

    def __init__(self, name=None, shape=scalar_shape):
        self._shape = shape
        self.name = name

    @property
    def shape(self):
        return self._shape

    def symbols(self):
        return {self}

    def __hash__(self):
        return hash(id(self))

    def __cmp__(self, other):
        return id(self) == id(other)


_t = Variable('t')


class Parameter(Algebraic):
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

        uid = (id(block), index)
        try:
            return Parameter._table[uid]
        except KeyError:
            pass
        obj = Algebraic.__new__(cls)
        obj.__init__()
        setattr(obj, 'uid', uid)
        setattr(obj, 'index', index)
        setattr(obj, '_parent', weakref.ref(block))
        Parameter._table[uid] = obj
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
        return self._parent().parameters[self.index]

    @property
    def shape(self):
        return scalar_shape

    def symbols(self):
        return {self}

    @staticmethod
    def from_block(block):
        return [Parameter(block, i) for i in range(len(block.parameters))]


@register_op()
def evaluate_signal(signal, t):
    return signal(t)


class Function(Algebraic):
    def __init__(self, shape, function, arguments):
        self._shape = shape
        self.function = function
        self.arguments = arguments

    @property
    def shape(self):
        return self._shape

    def __hash__(self):
        return hash((id(self.function), self.arguments))

    def symbols(self):
        return set(self.arguments)

    def __call__(self, *args):
        return self.function(*args)

    def call(self, args_dict):
        args = [args_dict[arg] for arg in self.arguments]
        return self.function(*args)


class SignalReference(Algebraic):
    """Symbolic variable representing a time varying signal.

    Args:
        port: The model port from which this signal is derived.

    """

    _signals = {}

    def __init__(self, port):
        self.port = port
        self._context = None

    def __new__(cls, port):
        source_id = id(port)
        try:
            new_signal = SignalReference._signals[source_id]()
            assert new_signal is not None
            return new_signal
        except (KeyError, AssertionError):
            pass
        new_signal = super().__new__(cls)
        SignalReference._signals[source_id] = weakref.ref(new_signal)
        return new_signal

    @property
    def t(self):
        if not self._context:
            return get_time_variable()
        else:
            return self._context.t

    @property
    def shape(self):
        return len(self.port),

    def __hash__(self):
        return hash(self.port)

    def __cmp__(self, other):
        try:
            return self.port is other.port
        except AttributeError:
            return False

    def __call__(self, t):
        if t is get_time_variable():
            return self
        try:
            self._context = t.context
        except AttributeError:
            pass
        return ExpressionGraph(evaluate_signal, self, t)

    def symbols(self):
        return {self, self.t}

    def call(self, args):
        function = args[self]
        result = Function(self.shape, function, [self.t])
        if self.t in args:
            return result.call(args)
        else:
            return result


def as_vector(arg):
    try:
        len(arg)
        return arg
    except TypeError:
        if isinstance(arg, (int, float)):
            return arg,
    if is_symbolic(arg):
        return backend.cast(arg)

    raise NotImplementedError(
        f'Don\'t know to to vectorise {arg.__class__}'
    )


def get_time_variable():
    return _t


def _is_subtree_constant(graph, node):
    obj = graph.nodes[node]
    if not is_op(obj):
        return not is_temporal(obj)
    if obj is evaluate_signal:
        return True
    return all(
        _is_subtree_constant(graph, child) for child in graph.edges[node]
    )


def is_temporal(symbol):
    if isinstance(symbol, ExpressionGraph):
        return not _is_subtree_constant(symbol, symbol.head)
    if isinstance(symbol, SignalReference):
        return True
    if symbol is get_time_variable():
        return True
    if is_op(symbol):
        return False
    return False


def is_matrix(obj):
    return isinstance(obj, (np.ndarray, spmatrix))


def lambdify(graph: ExpressionGraph,
             arguments: List[Union[Algebraic, List[Algebraic]]],
             name: str = 'f'
             ):

    substitutions = {}
    for i, arg in enumerate(arguments):
        if isinstance(arg, list):
            assert all(sub_arg.shape == scalar_shape for sub_arg in arg), \
                'Invalid arguments, lists must be a list of scalars'
            symbol = SymbolicVector(f'x_{i}', len(arg))
            substitutions.update(
                {sub_arg: symbol[j] for j, sub_arg in enumerate(arg)})
        else:
            try:
                n,  = arg.shape
            except ValueError as ex:
                n, m = arg.shape
                if m > 1:
                    raise ex
            symbol = SymbolicVector(f'x_{i}', n)
            substitutions[arg] = symbol

    def recurse(node):
        obj = graph.nodes[node]
        if is_op(obj):
            args = [recurse(child) for child in graph.edges[node]]
            return obj(*args)
        if is_matrix(obj):
            return backend.cast(obj)
        try:
            return substitutions[obj]
        except (KeyError, TypeError):
            return obj

    symbolic_expressions = recurse(graph.head)
    return backend.lambdify(
        symbolic_expressions, list(substitutions.values()), name
    )


class Quadrature(Algebraic):
    def __init__(self, integrand, context):
        self.integrand = integrand
        self._context = weakref.ref(context)
        self.index = context.add_quadrature(integrand)

    @property
    def shape(self):
        return self.integrand.shape

    @property
    def context(self):
        return self._context()

    def __hash__(self):
        return id(self)

    def symbols(self):
        return self.integrand.symbols()

    def __call__(self, t, *args):
        f = self.context.get_symbolic_integrator(self.context.parameters)
        return f(t, *args)


@register_op()
def time_integral(integrand, context):
    return Quadrature(integrand, context)


def concatenate(*arguments):
    length = 0
    constants = {}
    variables = {}

    for arg in arguments:
        if isinstance(arg, (int, float, complex)):
            constants[length] = arg
            length += 1
            continue

        assert arg.shape == scalar_shape, 'Cannot concatenate matrices'
        n, = arg.shape
        variables[arg] = list(range(length, length + n))
        length = length + n

    inclusions = []
    if constants:
        constant_indices, constant_vector = zip(*constants.items())
        inclusion = projection_matrix(constant_indices, length).T
        vector = np.array(constant_vector)
        inclusions.append((inclusion, vector))

    for variable, indicies in variables.items():
        inclusion_v = projection_matrix(indicies, length).T
        inclusions.append((inclusion_v, variable))

    pair = inclusions.pop()
    assert pair is not None

    result = pair[0] @ pair[1]
    while inclusions:
        pair = inclusions.pop()
        result += pair[0] @ pair[1]

    return result

