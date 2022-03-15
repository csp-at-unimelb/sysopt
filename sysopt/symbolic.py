from sysopt.backends import *


def projection_matrix(indices, dimension):
    matrix = sparse_matrix((len(indices), dimension))
    for i, j in indices:
        matrix[i, j] = 1

    return matrix


