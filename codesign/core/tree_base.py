from typing import NewType, Union
import numpy as np
from numbers import Number
import abc
from dataclasses import dataclass

Numeric = NewType('Numeric', Union[Number, np.ndarray])


def compare(a, b) -> bool:
    if a is b:
        return True
    raise NotImplementedError(f"cannot compare {a} and {b}")


class CannotBeChildException(Exception):
    pass


def validate_binop(f):
    invalid_ops = {
        '==': CannotBeChildException,
        '<=': CannotBeChildException,
        '>=': CannotBeChildException,
        '<': NotImplementedError,
        '>': NotImplementedError
    }

    def _f(x, y):
        for a in (x, y):
            if hasattr(a, 'op') and a.op in invalid_ops:
                raise invalid_ops[a.op]()
        return f(x, y)
    return _f


def equal(lhs, rhs):
    return ExpressionNode('==', lhs, rhs)


class UniqueObject(metaclass=abc.ABCMeta):

    def __hash__(self):
        return id(self)

    def __cmp__(self, other):
        return id(self) == id(other)


class Algebraic(metaclass=abc.ABCMeta):
    roots = {'==', '<=', '>='}

    def __add__(self, other):
        return ExpressionNode(Ops.add, self, other)

    def __mul__(self, other):
        return ExpressionNode(Ops.mul, self, other)

    def __rmul__(self, other):
        return ExpressionNode(Ops.mul, other, self)

    def __radd__(self, other):
        return ExpressionNode(Ops.add, other, self)

    def __sub__(self, other):
        return ExpressionNode(Ops.sub, self, other)

    def __rsub__(self, other):
        return ExpressionNode(Ops.sub, other, self)

    def __matmul__(self, other):
        return ExpressionNode(Ops.matmul, self, other)

    def __rmatmul__(self, other):
        return ExpressionNode(Ops.matmul, other, self)

    @validate_binop
    def __le__(self, other):
        return ExpressionNode('<=', self, other)

    @validate_binop
    def __ge__(self, other):
        return ExpressionNode('<=', other, self)

    @validate_binop
    def __eq__(self, other):
        return ExpressionNode('==', other, self)

    def __pow__(self, power, modulo=None):
        if modulo:
            raise NotImplementedError('Modulo not implemented')

        return ExpressionNode(Ops.power, self, power)

    def __neg__(self):
        return ExpressionNode(Ops.mul, -1, self)

    def __bool__(self):
        raise TypeError("Truth value of an expression not known")

    def __float__(self):
        raise TypeError

    def __int__(self):
        raise TypeError


class Ops:
    add = '+'
    sub = '-'
    mul = "*"
    matmul = '@'
    dot = 'dot'
    power = 'pow'
    integral = 'integral'
    eq = '=='
    leq = '<='
    cls = str


def is_symmetric(op: Ops.cls):
    return op in {Ops.add, Ops.eq, Ops.mul, Ops.dot}


class ExpressionNode(Algebraic):
    def __init__(self,
                 op: Ops.cls,
                 lhs: Union[Algebraic, Number, 'ExpressionNode'],
                 rhs: Union[Algebraic, Number, 'ExpressionNode']):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __hash__(self):
        return hash((hash(self.op), hash(self.lhs), hash(self.rhs)))

    def __eq__(self, other):
        return ExpressionNode(Ops.eq, self, other)

    @property
    def children(self):
        return self.lhs, self.rhs

    # def __cmp__(self, other):
    #     try:
    #         if self.op != other.op:
    #             return False
    #
    #         lhs, rhs = other.children
    #     except AttributeError:
    #         return False
    #     result = compare(self.lhs, lhs) and compare(self.rhs, rhs)
    #     if is_symmetric(self.op):
    #         result = result or (
    #                 compare(self.lhs, rhs) and compare(self.rhs, lhs)
    #         )
    #     return result

    def atoms(self):
        atoms = set()
        for node in (self.lhs, self.rhs):
            try:
                atoms |= node.atoms()
            except AttributeError:
                pass
        return atoms

    def is_root(self):
        return self.op in Algebraic.roots


def compare_expressions(expr, candidate):
    if isinstance(candidate, bool):
        try:
            return bool(expr) == candidate
        except TypeError:
            return False

    try:
        if candidate.op != expr.op:
            return False
    except AttributeError:
        return False

