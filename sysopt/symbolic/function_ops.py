"""Operations on multivariate vector functions."""

from abc import ABC, abstractmethod
from copy import copy
from typing import Union

from sysopt.types import Domain
from sysopt.symbolic.op_decorators import (
    require_equal_domains, require_equal_order_codomain
)


class FunctionOp(ABC):
    r"""Base class for Operations on functions.

    Functions $f$ are assumed to be mappings from

    math::
        f:T\times X \times Z \times U \times P \rightarrow R^n

    where
    - T is scalar
    - X, Z, U, P are (possibly empty) vectors
    - R^n is n-dimensional vector

    """
    domain: Domain
    codomain: Union[int, Domain]

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @require_equal_domains
    def __sub__(self, other):
        return Subtract(self, other)


class Concatenate(FunctionOp):
    """Concatenate functions of with the same domain"""
    def __init__(self, *args: FunctionOp):
        op, *remainder = args
        self.__vectors = [op]
        self.domain = op.domain.copy()
        self.codomain = op.codomain
        for arg in remainder:
            self.append(arg)

    @require_equal_domains
    @require_equal_order_codomain
    def append(self, other):
        self.__vectors.append(other)
        self.codomain += other.codomain

    def __call__(self, *args):
        calls = [f(*args) for f in self.__vectors]
        result = []
        for value in calls:
            if isinstance(value, list):
                result += value
            else:
                result.append(value)
        return result


class TensorProjection(FunctionOp):
    """Evaluate the projection on the given space."""
    def __init__(self,
                 domain: Domain,
                 variable: int,
                 index: int):
        self.codomain = 1
        self.domain = domain
        self.index = index
        self.variable = variable

    def __call__(self, *args):
        return args[self.variable][self.index]


class VectorProjection(FunctionOp):
    """Vector projection operation."""
    def __init__(self, domain, index):
        self.codomain = 1
        self.domain = domain
        self.index = index

    def __call__(self, *vector):
        return vector[self.index]


def project(domain, *args):
    if isinstance(domain, int):
        index, = args
        return VectorProjection(domain, index)
    else:
        variable, index = args
        space = Domain.index_of_field(variable)
        assert space >= 0
        return TensorProjection(domain, space, index)


class Subtract(FunctionOp):
    """Subtract the result of two functions with the same domain."""
    def __init__(self, lhs: FunctionOp, rhs: FunctionOp):
        assert lhs.domain == rhs.domain
        assert lhs.codomain == rhs.codomain
        self.lhs = lhs
        self.rhs = rhs
        self.domain = copy(lhs.domain)
        self.codomain = copy(lhs.codomain)

    def __call__(self, *args):
        return self.lhs(*args) - self.rhs(*args)


def subtract(lhs, rhs):
    return Subtract(lhs, rhs)


class Compose(FunctionOp):
    """Function composition."""
    def __init__(self, outer, inner):

        if inner.codomain != outer.domain:
            msg = f'Cannot compose {outer} with {inner}. '\
                  f'Inner has codomain {inner.codomain}, '\
                  f'while outer has domain {outer.domain}'
            raise TypeError(msg)

        self.inner = inner
        self.outer = outer

    @property
    def domain(self):
        return self.inner.domain

    @property
    def codomain(self):
        return self.outer.codomain

    def __call__(self, *args):
        r = self.inner(*args)
        res = self.outer(*r)
        return res


def compose(*args):
    outer, inner, *remainder = args
    if not remainder:
        return Compose(outer, inner)
    else:
        return compose(Compose(outer, inner), *remainder)
