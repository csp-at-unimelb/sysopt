from typing import Tuple
import numpy as np


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
            elif self.shape == (1, 1):
                return self[0, 0] == other
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


def sparse_matrix(shape: Tuple[int, int]):
    return np.zeros(shape, dtype=float).view(Matrix)


def basis_vector(index, dimension):
    e_i = np.zeros(shape=(dimension, ), dtype=float).view(Matrix)
    e_i[index] = 1
    return e_i

