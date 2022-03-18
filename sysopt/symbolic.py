"""Interface for Symbolic Functions and AutoDiff."""
from abc import ABC
# pylint: disable=wildcard-import,unused-wildcard-import
from sysopt.backends import *


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


class Symbolic(ABC):
    pass


Symbolic.register(SymbolicVector)
