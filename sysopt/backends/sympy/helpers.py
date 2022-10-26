"""Sympy helper functions"""
import sympy
from typing import Tuple


def sympy_vector(name: str, shape=Tuple[int, ...]):
    if len(shape) == 1:
        return sympy.Matrix([sympy.symbols(f'{name}_{i}')
                             for i in range(shape[0])])
    elif len(shape) == 2:
        return sympy.Matrix(
            [[sympy.symbols(f'{name}_{i,j}')
              for j in range(shape[1])]
             for i in range(shape[0])])

    raise NotImplementedError
