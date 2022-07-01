"""Functions and factories to create symbolic variables."""
import weakref
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from inspect import signature

import numpy as np
from typing import Union, List, Callable, Tuple, Optional, Dict, NewType, Any

from sysopt.helpers import flatten
from sysopt.exceptions import InvalidShape, EvaluationError
array = np.array
epsilon = 1e-12

SymbolicAtom = NewType('SymbolicAtom', Union['Variable', 'Parameter'])
SymbolicArray = NewType(
    'SymbolicArray', Union[List[SymbolicAtom], Tuple[SymbolicAtom]]
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


class Matrix(np.ndarray):
    """View of a numpy matrix for use in expression graphs."""

    def __hash__(self):
        shape_hash = hash(self.shape)
        data_hash = hash(tuple(self.ravel()))
        return hash((shape_hash, data_hash))

    def __cmp__(self, other):
        return self is other

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return other == self.tolist()

        if isinstance(other, (float, int, complex)):
            if self.shape == (1,):
                return self[0] == other
            elif self.shape == (1,1):
                return self[0,0] == other
            return False

        try:
            if self.shape != other.shape:
                return False
        except AttributeError:
            return False
        if hash(self) != hash(other):
            return False
        result = (self - other) == 0
        if isinstance(result, np.ndarray):
            return result.all()
        else:
            return result

def as_array(
    item:Union[List[Union[int, float]], int, float, np.ndarray],
    prototype:SymbolicAtom =None):
    if isinstance(item, Algebraic):
        return item
    if isinstance(item, np.ndarray):
        return item.view(Matrix)
    elif isinstance(item, (list, tuple)):
        return concatenate(*item)
    elif isinstance(item, (int, float)):
        m = np.array([item],dtype=float).view(Matrix)
        return m
    elif item is None:
        return None
    elif callable(item):
        return Function(prototype.shape, item, prototype)
    raise NotImplementedError(
        f'Don\'t know how to treat {item} as an array')


def sparse_matrix(shape: Tuple[int, int]):
    return np.zeros(shape, dtype=float).view(Matrix)


numpy_handlers = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        numpy_handlers[numpy_function] = func
        return func
    return decorator


def inclusion_map(basis_map: Dict[int, int],
                  domain_dimension: int,
                  codomain_dimension: int):
    """Project the domain onto a subspace spanned by the indices.

    row space: dimension of the subspace
    col space: dimension of the source

    Args:
        basis_map:          Coordinate indices of the vector subspace.
        domain_dimension:   Dimension of the domain.
        codomain_dimension: Dimension of the codomain

    basis map should be of the form domain -> codomain

    """

    if domain_dimension == 1:
        if codomain_dimension == 1:
            return lambda x: ExpressionGraph(mul, 1, x)
        else:
            assert len(basis_map) == 1
            j = basis_map[0]
            e_j = basis_vector(j, codomain_dimension)
            return lambda x: ExpressionGraph(mul, e_j, x)

    matrix = sparse_matrix((codomain_dimension, domain_dimension))

    for i, j in basis_map.items():
        matrix[j, i] = 1
    return LinearMap(matrix)


def basis_vector(index, dimension):
    e_i = np.zeros(shape=(dimension, ), dtype=float).view(Matrix)
    e_i[index] = 1
    return e_i

class LinearMap:
    """Functional representation of a linear operator defined by a matrix."""


    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def T(self):    # pylint: disable=invalid-name
        return LinearMap(self.matrix.T)

    def __call__(self, arg):
        return ExpressionGraph(matmul, self.matrix, arg)


def restriction_map(indices: Union[List[int], Dict[int, int]],
                    superset_dimension: int) -> Callable:

    if isinstance(indices, (list, tuple)):
        domain = len(indices)
        iterator = enumerate(indices)
    else:
        domain = max(indices.keys()) + 1
        iterator = indices.items()

    if domain == superset_dimension == 1:
        return lambda x: ExpressionGraph(mul, 1, x)

    matrix = sparse_matrix((domain, superset_dimension))
    for i, j in iterator:
        matrix[i, j] = 1

    return LinearMap(matrix)


__ops = defaultdict(list)
__shape_ops = {}
__op_strings = {}
scalar_shape = (1, )


def infer_scalar_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    this_shape = shapes[0]
    for shape in shapes[1:]:
        if shape in (this_shape, scalar_shape):
            continue
        if this_shape == (1, ):
            this_shape = shape
        else:
            raise InvalidShape('Invalid Shape')
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
            raise InvalidShape('Invalid shape')
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


def register_op(shape_func=infer_scalar_shape, string=None):
    """Decorator which register the operator as an expression graph op."""
    def wrapper(func):
        sig = signature(func)
        is_variable = any(
            param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            for param in sig.parameters.values())

        idx = None if is_variable else len(sig.parameters)
        __ops[idx].append(func)
        __shape_ops[func] = shape_func
        if string:
            __op_strings[func] = string
        return func

    return wrapper

def op_to_string(op):
    try:
        return __op_strings[op]
    except KeyError:
        return str(op)

def wrap_as_op(func: Callable,
               arguments: Optional[int] = None,
               shape_func=infer_scalar_shape,
               numpy_func=None
               ) -> Callable:
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

    if numpy_func is not None:
        numpy_handlers[numpy_func] = wrapper

    return wrapper


@register_op(string='pow')
@implements(np.power)
def power(base, exponent):
    return base ** exponent



@register_op(string='+')
def add(lhs, rhs):
    return lhs + rhs


@register_op(string='-')
def sub(lhs, rhs):
    return lhs - rhs

def is_scalar(item):
    if isinstance(item, (float, int, complex)):
        return True
    try:
        return item.shape == scalar_shape
    except AttributeError:
        pass
    return False

@register_op(shape_func=matmul_shape, string='@')
def matmul(lhs, rhs):
    if isinstance(lhs, (int, float)) or isinstance(rhs, (int, float)):
        return lhs * rhs

    return lhs @ rhs


@register_op(string='-')
def neg(obj):
    return -obj


@register_op(string='*')
def mul(lhs, rhs):
    return lhs * rhs


@register_op(string='/')
def div(lhs, rhs):
    return lhs / rhs


@register_op(shape_func=transpose_shape, string='transpose')
def transpose(matrix):
    return matrix.T


def slice_to_list(slce: slice, max_len=None):

    n = slce.stop or max_len
    return list(range(n))[slce]


class Inequality:
    """Inequality expression.

    Non-negative evaluation means that the inequality is satisfied.

    """
    def __init__(self, smaller, bigger):
        self.smaller = smaller
        self.bigger = bigger

    def __repr__(self):
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
    """
        Non-negative evaluation means that the inequality is satisfied.

    """
    def to_ode(self, regulariser, alpha=1):
        """
        Implements the K-S functional
        math::
            c(g,rho) = ln(int_0^t exp[- rho*g ]dt/alpha)/rho

        Args:
            regulariser:
            alpha:      Weighting constant

        Returns:
            c, dc/dt - where c is the variable and dc/dt is an expression graph
            for the dynamics. Where c < zero as rho-> infity implies
            the constraint is violated.

        """
        c = Variable('c')
        rho = regulariser
        g = self.to_graph()
        # TODO: fix me
        # pylint: disable=import-outside-toplevel
        from sysopt.symbolic.scalar_ops import exp

        return c, exp(rho * (c - g)) / (alpha * rho)

def is_zero(arg, shape=scalar_shape):
    try:
        r = bool(arg == 0)
        return r
    except ValueError:
        pass
    try:
        return arg.shape == shape and (arg == 0).all()
    except AttributeError:
        pass
    return False


class Algebraic(metaclass=ABCMeta):
    """Base class for symbolic terms in expression graphs."""

    def __len__(self):
        if len(self.shape) == 1:
            return self.shape[0]

        raise ValueError('Cannot determine the length of a non-vector object')

    def __array_ufunc__(self, func, method, *args, **kwargs):
        if func not in numpy_handlers:
            return NotImplemented
        if method != '__call__':
            return NotImplemented
        return numpy_handlers[func](*args, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in numpy_handlers:
            return NotImplemented
        return numpy_handlers[func](*args, **kwargs)

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(
            f'{str(self.__class__)} is missing this method'
        )

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    def T(self):  # pylint: disable=invalid-name
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
            indices = slice_to_list(item, n)
        else:
            indices = [item]

        if indices == [range(n)]:
            return self

        pi = restriction_map(indices, n)
        return pi(self)

    def __iter__(self):
        n = self.shape[0]
        for i in range(n):
            yield restriction_map([i], n)(self)

    def __add__(self, other):
        if is_zero(other, self.shape):
            return self

        return ExpressionGraph(add, self, other)

    def __radd__(self, other):
        if is_zero(other, self.shape):
            return self
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


def recursively_apply(graph: 'ExpressionGraph',
                      trunk_function,
                      leaf_function,
                      current_node=None):

    if isinstance(graph, ConstantFunction):
        return leaf_function(graph.value)
    if isinstance(graph, GraphWrapper):
        return recursively_apply(graph.graph, trunk_function, leaf_function)

    sorted_nodes = graph.get_topological_sorted_indices()
    trunk_indices = {i for i in sorted_nodes if i in graph.edges}
    context = {}
    while sorted_nodes:
        node = sorted_nodes.pop()
        if node in context:
            continue
        item = graph.nodes[node]
        if node in trunk_indices:
            args = [context[i] for i in graph.edges[node]]
            context[node] = trunk_function(item, *args)
        else:
            context[node] = leaf_function(item)

    return context[graph.head]



def match_args_by_name(expr: Algebraic, arguments: Dict[str, Any]):
    return {
        a: arguments[a.name] for a in expr.symbols()
        if hasattr(a, 'name') and a.name in arguments
    }


class ExpressionGraph(Algebraic):
    """Graph stresentation of a symbolic expression."""

    def __init__(self, op, *args):
        self.nodes = []
        self.edges = defaultdict(list)
        self._head = None
        op_node = self.add_or_get_node(op)
        self.edges.update(
            {op_node: [self.add_or_get_node(a) for a in args]}
        )
        self._shape = None
        self.head = op_node

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._shape = self._get_shape_of(value)
        self._head = value

    @property
    def shape(self):
        return self._shape

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

    def __call__(self, *args, **kwargs):

        assert len(args) == len(self.symbols()),\
            f'Tried to call function with {self.symbols()} '
        values = dict(zip(self.symbols(), args))
        return self.call(values)

    def call(self, values):
        invalid_args = {
            str(k):v for k,v in values.items() if v is None
        }
        if invalid_args:
            raise TypeError(f'Invalid arguments {invalid_args}')
        arugments = {
            k: as_array(v, prototype=k) for k,v in values.items()
        }

        context = {}
        sorted_nodes = self.get_topological_sorted_indices()

        def eval_node(obj):
            if is_op(obj):
                args = [context[child] for child in self.edges[node]]
                out = obj(*args)

                return out
            try:
                return obj.call(arugments)
            except (AttributeError, TypeError):
                pass
            try:
                return arugments[obj]
            except (KeyError, TypeError):
                pass
            return obj
        path = []
        while sorted_nodes:
            node = sorted_nodes.pop()
            path.append(node)
            if node in context:
                continue
            obj = self.nodes[node]
            try:
                context[node] = eval_node(obj)
            except Exception as ex:
                raise EvaluationError(self, context, path, ex) from ex
        result = context[self.head]
        try:
            # mostly for numpy arrays, which we assume are
            # the most common thing passing through here.
            return result.reshape(self.shape)
        except (AttributeError, TypeError, NotImplementedError):
            return result

    @property
    def is_symbolic(self):
        return self.symbols() != {}

    def add_or_get_node(self, value):
        if value is None:
            raise TypeError('unsupported operand for \'NoneType\'')
        if value is self:
            assert self.head is not None
            return self.head

        if isinstance(value, ExpressionGraph):
            return self.merge_and_return_subgraph_head(value)

        idx = len(self.nodes)
        if is_op(value):
            self.nodes.append(value)
            return idx
        # else - a scalar or matrix
        try:
            # already in the array
            return self.nodes.index(value)
        except ValueError:
            pass

        self.nodes.append(as_array(value))
        return idx

    def merge_and_return_subgraph_head(self, other):
        new_indices = {
            old_idx: self.add_or_get_node(node)
            for old_idx, node in enumerate(other.nodes)
        }
        for parent, children in other.edges.items():
            new_source_idx = new_indices[parent]
            if not children:
                continue
            self.edges[new_source_idx] += [
                new_indices[child] for child in children
            ]
        return new_indices[other.head]

    def push_op(self, op, *nodes):
        op_node = self.add_or_get_node(op)
        node_indices = [self.add_or_get_node(node) for node in nodes]
        self.edges[op_node] = node_indices
        self.head = op_node
        assert self.shape

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
        edge_list = sorted(list(
            (parent, child) for parent, children in self.edges.items()
            for child in children
        ))
        edge_hash = hash(tuple(edge_list))

        def hash_nodes():
            hashes = []
            for node in self.nodes:
                hashes.append(hash(node))
            return hashes

        return hash((edge_hash, *hash_nodes()))

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

    def get_subtree_at(self, index):

        def recurse(node_idx):
            if node_idx not in self.edges:
                return self.nodes[node_idx]
            else:
                return ExpressionGraph(
                    self.nodes[node_idx],
                    *[recurse(idx) for idx in self.edges[node_idx]]
                )

        return recurse(index)

    def list_subtree(self, index):
        visited = set()
        unvisited = set(index)
        while unvisited:
            item = unvisited.pop()
            if item in visited:
                continue
            visited.add(item)
            if item in self.edges:
                unvisited |= set(self.edges[item])

        return visited

    def get_topological_sorted_indices(self):
        """Topological sort via Kahn's algorithm."""

        # Tree is organised as:
        #
        #         parent
        #         /   \
        #      child  child
        #
        # edges point from parent to child

        frontier = {self.head}

        edges = {i: l.copy() for i, l in self.edges.items() if l}
        reverse_graph = defaultdict(list)

        for in_node, out_nodes in edges.items():
            for out_node in out_nodes:
                reverse_graph[out_node].append(in_node)
        result = []
        while frontier:
            node = frontier.pop()
            result.append(node)
            if node not in edges:
                continue
            while edges[node]:
                child = edges[node].pop()
                reverse_graph[child].remove(node)
                if not reverse_graph[child]:
                    frontier.add(child)

        if any(lst != [] for lst in edges.values()):
            raise ValueError(f'Graph has cycles: {edges}')
        return result

    def is_acyclic(self):
        self.get_topological_sorted_indices()
        return True

    def __repr__(self):
        # assert self.is_acyclic()

        def trunk_function(node_object, *children):
            args = ','.join(children)

            string = op_to_string(node_object)

            return f'({string}: {args})'

        return recursively_apply(self, trunk_function, str)


numpy_handlers.update(
    {
        np.matmul: lambda a, b: ExpressionGraph(matmul, a, b),
        np.multiply: lambda a, b: ExpressionGraph(mul, a, b),
        np.add: lambda a, b: ExpressionGraph(add, a, b),
        np.subtract: lambda a, b: ExpressionGraph(sub, a, b),
        np.divide: lambda a, b: ExpressionGraph(div, a, b),
        np.negative: lambda x: ExpressionGraph(neg, x),
        np.transpose: lambda x: ExpressionGraph(transpose, x),
        np.power: lambda a, b: ExpressionGraph(power, a, b),
        np.deg2rad: lambda x: ExpressionGraph(mul, np.pi/180, x),
        np.rad2deg: lambda x: ExpressionGraph(mul, 180/np.pi, x)
    }
)


def symbolic_vector(name, length=1):
    return Variable(name, shape=(length, ))


class Variable(Algebraic):
    """Symbolic type for a free variable."""
    is_symbolic = True
    __array_ufunc__ = None

    def __init__(self, name=None, shape=scalar_shape):
        self._shape = shape
        self.name = name

    def __str__(self):
        shape_str = 'x'.join(str(n) for n in self.shape)
        return f'{self.name}^({shape_str})'

    def __repr__(self):
        if self.name is not None:
            return f'{self.__class__.__name__}({self.name}, {self.shape})'
        else:
            return 'unnamed_variable'

    @property
    def shape(self):
        return self._shape

    def symbols(self):
        return {self}

    def __hash__(self):
        return hash(id(self))

    def __cmp__(self, other):
        return id(self) == id(other)


_t = Variable('time')


def resolve_parameter_uid(block, index):
    name = block.parameters[index]
    return hash(name)


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

        uid = resolve_parameter_uid(block, index)

        try:
            obj = Parameter._table[uid]
            return obj
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


@register_op()
def evaluate_signal(signal, t):
    return signal(t)


class Function(Algebraic):
    """Wrapper for function calls."""

    def __init__(self, shape, function, arguments):
        self._shape = shape
        self.function = function
        self.arguments = tuple(arguments)

    def __repr__(self):
        args = ','.join(str(a) for a in self.arguments)
        return f'{str(self.function)}({args})'

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
        result = self.function(*args)
        return result


class SignalReference(Algebraic):
    """Symbolic variable stresenting a time varying signal.

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

    def __repr__(self):
        return str(self.port)

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
        if isinstance(function, ExpressionGraph):
            result = function
        else:
            result = Function(self.shape, function, [self.t])

        if self.t in args:
            return result.call(args)
        else:
            return result


def list_symbols(arg):
    return arg.symbols()


def is_symbolic(arg):
    if isinstance(arg, list):
        return any(is_symbolic(a) for a in arg)
    try:
        return arg.is_symbolic
    except AttributeError:
        return False


def is_vector_like(arg):
    if is_matrix(arg):
        return len(arg.shape) == 1 or (
            len(arg.shape) == 2 and arg.shape[1] == 1)
    return False


def as_vector(arg):
    if isinstance(arg, Algebraic) or is_vector_like(arg):
        return arg
    if isinstance(arg, (list, tuple)):
        return flatten(arg, 1)

    if isinstance(arg, (int, float)):
        return [arg]

    if arg is None:
        return []

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
    if isinstance(symbol, PathInequality):
        return True
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
    return isinstance(obj, np.ndarray)


class Quadrature(Algebraic):
    """Variable representing a quadrature."""

    def __init__(self, integrand, context):
        self.integrand = integrand
        self._context = weakref.ref(context)
        self.index = context.add_quadrature(integrand)

    def __repr__(self):
        return f'(int_0^t {str(self.integrand)} dt'

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
        return self.context.evaluate_quadrature(self.index, t, *args)


def concatenate(*arguments):
    length = 0
    scalar_constants = []
    vectors = []
    # put all scalar constants into a single vector
    # multiply all vector constants and vector graphs by inclusion maps
    if all(isinstance(a, np.ndarray) for a in arguments):
        return np.concatenate(arguments)

    for arg in arguments:
        if arg is None:
            continue
        if isinstance(arg, (int, float, complex)):
            scalar_constants.append((length, arg))
            length += 1
            continue
        assert len(arg.shape) == 1,\
            f'Cannot concatenate object with shape {arg.shape}'

        n, = arg.shape
        basis_map = dict(enumerate(range(length, length + n)))
        vectors.append((basis_map, n, arg))
        length = length + n

    result = sparse_matrix((length, ))
    while scalar_constants:
        i, v = scalar_constants.pop()
        result[i] = v

    while vectors:
        basis_map, domain, vector = vectors.pop()
        inclusion = inclusion_map(basis_map, domain, length)
        result = result + inclusion(vector)

    return result


def extract_quadratures(graph: ExpressionGraph) \
        -> Tuple[ExpressionGraph, Dict[Variable, ExpressionGraph]]:

    quadratures = {}

    def recurse(node_idx):
        node = graph.nodes[node_idx]
        if isinstance(node, Quadrature):
            q = Variable('q')
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

    return out_graph, quadratures


def create_log_barrier_function(constraint, stiffness):
    # TODO: fix me
    # pylint: disable=import-outside-toplevel
    from sysopt.symbolic.scalar_ops import log
    return - stiffness * log(stiffness * constraint + 1)


class ConstantFunction(Algebraic):
    """Wrap a constant value and treat it like a function."""
    def __init__(self, value, arguments: List[Union[Variable, Parameter]]):
        if isinstance(value, np.ndarray):
            self.value = value.view(Matrix)
        elif isinstance(value, (list, tuple)):
            self.value = np.array(value).view(Matrix)
        else:
            self.value = as_array(value)

        self.arguments = arguments

    def __hash__(self):
        return hash((*self.arguments, self.value))

    @property
    def shape(self):
        if isinstance(self.value, (float, int)):
            return scalar_shape
        else:
            return self.value.shape

    def __repr__(self):
        return str(self.value)

    def symbols(self):
        return self.arguments

    def __call__(self, *args, **kwargs):
        return self.value


class GraphWrapper(Algebraic):
    """Wraps an expression graph with the specified arguments."""

    def __init__(self, graph:ExpressionGraph, arguments:List[SymbolicAtom]):
        self.arguments = tuple(arguments)
        unbound_symbols = {
            s for s in graph.symbols() if s not in arguments
        }
        if unbound_symbols:
            raise ValueError('Could not create function from graph due to'
                             f'unbound symbolic variables: {unbound_symbols}')
        self.graph = graph

    def symbols(self):
        return set(self.arguments)

    @property
    def shape(self):
        return self.graph.shape

    def call(self, args:Dict[SymbolicAtom, Any]):

        symbols = self.graph.symbols()
        inner_args = {
            a: args[a] for a in self.arguments
            if a in symbols
        }
        return self.graph.call(inner_args)

    def __call__(self, *args):
        if len(args) != len(self.arguments):
            raise ValueError(
                f'Invalid arguments; expected {self.arguments}, '
                f'but recieved {args}')
        return self.call(dict(zip(self.arguments, args)))

    def __hash__(self):
        return hash((hash(self.graph), hash(tuple(self.symbols()))))

    def __repr__(self):
        return f'{self.symbols()} ->  {self.graph}'

def function_from_graph(graph: ExpressionGraph, arguments: List[SymbolicAtom]):
    if not isinstance(graph, ExpressionGraph):
        if graph is None:
            return None
        return ConstantFunction(graph, arguments)

    return GraphWrapper(graph, arguments)
