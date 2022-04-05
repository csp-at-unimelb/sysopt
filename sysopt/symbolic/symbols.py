"""Functions and factories to create symbolic variables."""
import weakref
from abc import ABCMeta, abstractmethod, ABC
from collections import defaultdict
from inspect import signature
from scipy.sparse import dok_matrix
from typing import Union

from sysopt.backends import cast
from sysopt.backends import is_symbolic as _is_symbolic
from sysopt.backends import list_symbols as _list_symbols


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


def sparse_matrix(shape):
    return dok_matrix(shape, dtype=float)


def is_symbolic(obj):
    try:
        return obj.is_symbolic
    except AttributeError:
        return _is_symbolic(obj)


def list_symbols(obj):
    try:
        return obj.symbols()
    except AttributeError:
        return _list_symbols(obj)


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in enumerate(indices):
        matrix[i, j] = 1

    return matrix


__ops = defaultdict(list)
__shape_ops = {}
__VAR = "variable"

scalar_shape = (1, )


def infer_scalar_shape(*shapes):
    this_shape = shapes[0]
    for shape in shapes[1:]:
        if this_shape == shape or shape == scalar_shape:
            continue
        if this_shape == (1, ):
            this_shape = shape
        else:
            raise AttributeError('Invalid Shape')
    return this_shape


def matmul_shape(*shapes):
    n, m = shapes[0]
    for n_next, m_next in shapes[1:]:
        if m != n_next:
            raise AttributeError("Invalid shape")
        else:
            m = n_next
    return n, m


def transpose_shape(shape):
    n, m = shape
    return m, n


def infer_shape(op, *shapes):
    return __shape_ops[op](*shapes)


def operation(shape_func=infer_scalar_shape):
    def wrapper(func):
        sig = signature(func)
        is_variable = any(
            param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            for param in sig.parameters.values())

        idx = __VAR if is_variable else len(sig.parameters)
        __ops[idx].append(func)
        __shape_ops[func] = shape_func
        return func

    return wrapper


def wrap_function(func,
                  n_arguments=__VAR,
                  shape_func=infer_scalar_shape):
    __ops[n_arguments].append(func)

    def wrapper(*args):
        return ExpressionGraph(func, *args)

    __shape_ops[func] = shape_func

    return wrapper


@operation()
def power(base, exponent):
    return base ** exponent


@operation()
def add(lhs, rhs):
    return lhs + rhs


@operation()
def sub(lhs, rhs):
    return lhs - rhs


@operation(matmul_shape)
def matmul(lhs, rhs):
    return lhs @ rhs


@operation()
def neg(obj):
    return -obj


@operation()
def mul(lhs, rhs):
    return lhs * rhs


@operation()
def div(lhs, rhs):
    return lhs / rhs


@operation(transpose_shape)
def transpose(matrix):
    return matrix.T


class Inequality:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'{self.lhs} {self.op} {self.rhs}'

    def symbols(self):
        try:
            result = self.lhs.symbols()
        except AttributeError:
            result = set()

        try:
            result |= self.rhs.symbols()
        except AttributeError:
            pass
        return result


class Algebraic(metaclass=ABCMeta):

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def symbols(self):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

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
        return Inequality('le', self, other)

    def __ge__(self, other):
        return Inequality('ge', self, other)

    def __gt__(self, other):
        return Inequality('gt', self, other)

    def __lt__(self, other):
        return Inequality('lt', self, other)

    def __cmp__(self, other):
        return id(self) == id(other)


def is_op(value):
    return any(value in ops for ops in __ops.values())


class ExpressionGraph(Algebraic):
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

    def __add__(self, other):
        return self.push_op(add, self, other)

    def __radd__(self, other):
        return self.push_op(add,  other, self)

    def __neg__(self):
        return self.push_op(neg, self)

    def __sub__(self, other):
        return self.push_op(sub, self, other)

    def __rsub__(self, other):
        return self.push_op(sub, other, self)

    def __mul__(self, other):
        return self.push_op(mul, self, other)

    def __rmul__(self, other):
        return self.push_op(mul, other, self)

    def __truediv__(self, other):
        return self.push_op(div, self, other)

    def __rtruediv__(self, other):
        return self.push_op(div, other,  self)

    def __matmul__(self, other):
        return self.push_op(matmul, self, other)

    def __rmatmul__(self, other):
        return self.push_op(matmul, other, self)

    def __pow__(self, exponent, modulo=None):
        return self.push_op(power, self, exponent)

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
    is_symbolic = True

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
    __table = {}

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
            return Parameter.__table[uid]
        except KeyError:
            pass
        obj = Algebraic.__new__(cls)
        obj.__init__()
        setattr(obj, 'uid', uid)
        setattr(obj, 'index', index)
        setattr(obj, '_parent', weakref.ref(block))
        Parameter.__table[uid] = obj
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


@operation()
def evaluate_signal(signal, t):
    return signal(t)


class SignalReference(Algebraic):
    t = _t

    def __init__(self, reference):
        self.reference = reference
        self._shape = (len(reference), )

    @property
    def shape(self):
        return self._shape

    def __hash__(self):
        return hash(self.reference)

    def __cmp__(self, other):
        try:
            return self.reference == other.reference
        except AttributeError:
            return False

    def __call__(self, t):
        return ExpressionGraph(evaluate_signal, self, t)

    def symbols(self):
        return {self, self.t}


def as_vector(arg):
    try:
        len(arg)
        return arg
    except TypeError:
        if isinstance(arg, (int, float)):
            return arg,
    if is_symbolic(arg):
        return cast(arg)

    raise NotImplementedError(
        f'Don\'t know to to vectorise {arg.__class__}'
    )


def get_time_variable():
    return _t


def _is_subtree_constant(graph, node):
    result = True
    obj = graph.nodes[node]
    if not is_op(obj):
        return not is_temporal(obj)
    if obj is evaluate_signal:
        return True
    return all(
        _is_subtree_constant(graph, child)
        for child in graph.edges[node]
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
