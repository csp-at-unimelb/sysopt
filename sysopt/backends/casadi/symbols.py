"""Casadi implementation of symbolic vector and helper functions."""

import casadi as _casadi
import numpy as np


class SymbolicVector(_casadi.SX):
    """Wrapper around SX for vectors."""
    _names = {}

    def __init__(self, *args, **kwarg):
        super().__init__()

    def __repr__(self):
        return self._name

    def __hash__(self):
        return id(self)

    @staticmethod
    def _validate_name(name):

        try:
            idx = SymbolicVector._names[name]
        except KeyError:
            SymbolicVector._names[name] = 0
            idx = 0

        SymbolicVector._names[name] += 1
        return f'{name}_{idx}'

    def __new__(cls, name, length=1):
        assert isinstance(length, int)
        obj = SymbolicVector.sym(name, length)
        obj._name = SymbolicVector._validate_name(name)
        obj.__class__ = cls
        if cls is not SymbolicVector:
            obj.__bases__ = [SymbolicVector, _casadi.SX]
        return obj

    @staticmethod
    def from_iterable(arg):
        n = len(arg)
        obj = SymbolicVector('x', n)
        for i in range(n):
            if isinstance(arg[i], _casadi.SX):
                obj[i] = arg[i]
            else:
                obj[i] = arg[i]
        return obj

    @staticmethod
    def from_DM(arg):  # pylint: disable=invalid-name
        m = None
        try:
            n, m = arg.shape
        except TypeError:
            n, = arg.shape

        assert not m or m == 1, \
            f'Cannot convert object with shape {arg.shape}'
        obj = SymbolicVector('x', n)
        for i in range(n):
            obj[i] = arg[i]
        return obj

    @staticmethod
    def from_sx(arg):
        m = None
        try:
            _, m = arg.shape
        except TypeError:
            pass

        assert not m or m == 1, \
            f'Cannot convert object with shape {arg.shape}'
        try:
            bases = list(set(arg.__bases__) | {arg.__class__})
        except AttributeError:
            bases = [arg.__class__]
        if not hasattr(arg, '_name'):
            setattr(arg, '_name', SymbolicVector._validate_name('x'))
        setattr(arg, '__class__', SymbolicVector)
        setattr(arg, '__bases__', bases)

        return arg

    def __iter__(self):
        return iter(
            [self[i] for i in range(self.shape[0])]
        )

    def __len__(self):
        return self.shape[0]

    def index(self, value):
        for i, v in enumerate(self):
            if v is value:
                return i
        return -1

    def __setitem__(self, key, value):
        if isinstance(key, slice) and isinstance(value, list):
            step = key.step or 1
            for i, j in enumerate(range(key.start, key.start, step)):
                super().__setitem__(j, value[i])
        else:
            super().__setitem__(key, value)

    def __eq__(self, other):
        if other is self:
            return True
        try:
            if len(other) == len(self):
                return all(i == j for i, j in zip(self, other))

        # Casadi likes to throw bare exceptions.
        except:  # pylint: disable=bare-except
            pass

        return super().__eq__(other)


def concatenate(*vectors):
    """Concatenate arguments into a casadi symbolic vector."""
    try:
        v0, *v_n = vectors
    except ValueError:
        return None
    while v0 is None:
        try:
            v0, *v_n = v_n
        except ValueError:
            return None
    if isinstance(v0, (tuple, list)):
        result = concatenate(*v0)
    else:
        result = cast(v0)
    if not isinstance(v0, SymbolicVector):
        result = v0
    for v_i in v_n:
        if v_i is not None:
            result = _casadi.vertcat(result, v_i)

    return cast(result)


def cast(arg):
    if arg is None:
        return None
    if isinstance(arg, (float, int)):
        return SymbolicVector.from_iterable([arg])
    if isinstance(arg, _casadi.SX):
        return SymbolicVector.from_sx(arg)
    elif isinstance(arg, (list, tuple, np.ndarray)):
        return SymbolicVector.from_iterable(arg)
    elif isinstance(arg, _casadi.DM):
        return SymbolicVector.from_DM(arg)

    raise NotImplementedError(f'Don\'t know how to cast {arg.__class__}')


def is_symbolic(arg):
    if isinstance(arg, (list, tuple)):
        return all(is_symbolic(item) for item in arg)
    if hasattr(arg, 'is_symbolic'):
        return arg.is_symbolic
    return isinstance(arg, _casadi.SX)


def constant(value):
    assert isinstance(value, (int, float))
    c = _casadi.SX(value)
    return c

