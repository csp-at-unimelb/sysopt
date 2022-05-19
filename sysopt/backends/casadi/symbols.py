"""Casadi implementation of symbolic vector and helper functions."""
import casadi as _casadi
import numpy as np
from scipy.sparse import dok_matrix
from sysopt.symbolic import casts


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
    for v_i in v_n:
        if v_i is not None:
            result = _casadi.vertcat(result, v_i)

    return cast(result)


def cast(arg):
    if arg is None:
        return None
    if isinstance(arg, np.ndarray):
        r = _casadi.DM_from_array(arg, check_only=False)
        return r

    return casts.cast_type(arg)


@casts.register((float, int), _casadi.SX)
def cast_scalar(arg):
    return _casadi.SX(arg)
    # return SymbolicVector.from_iterable([arg])


@casts.register(list, _casadi.SX)
def cast_iterable(arg):
    return _casadi.vertcat(
        *[casts.cast_type(a, _casadi.SX) for a in arg]
    )

#    return _casadi.SX(arg)


def is_symbolic(arg):
    if isinstance(arg, (list, tuple)):
        return all(is_symbolic(item) for item in arg)
    if hasattr(arg, 'is_symbolic'):
        return arg.is_symbolic
    return isinstance(arg, _casadi.SX)


def constant(value):
    assert isinstance(value, (int, float))
    c = _casadi.SX([value])
    return c


@casts.register(_casadi.DM, list)
def dm_to_list(var: _casadi.DM):
    n, m = var.shape
    the_array = var.toarray()
    the_list = [[the_array[i, j] for j in range(m)] for i in range(n)]
    return the_list


@casts.register(dok_matrix, _casadi.SX)
def sparse_matrix_to_sx(matrix):
    return _casadi.SX(matrix)


@casts.register(_casadi.SX, list)
def to_list(arg):
    if arg.shape[1] == 1:
        return [arg[i] for i in range(arg.shape[0])]
    raise NotImplementedError
