"""Interface for Symbolic Functions and AutoDiff."""

# pylint: disable=wildcard-import,unused-wildcard-import
from sysopt.backends import *


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


