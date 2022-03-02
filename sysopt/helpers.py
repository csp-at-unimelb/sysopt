import numpy as np


def lobatto_gauss_legendre_scheme(n, dtype=float, tolerance=1e-9):
    size = n + 1
    P = np.zeros((size, size), dtype=dtype)

    x = np.cos(np.pi * np.linspace(1, 0, size))
    x_old = 2 * np.ones_like(x)

    while max(abs(x -x_old)) > tolerance:
        x_old[:] = x
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(1, n):
            P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1])/(k + 1)

        x = x_old - (x * P[:, n] - P[:, n - 1]) / (size * P[:, n])

    weights = 2 / (size * n * P[:, n] * P[:, n])

    return x, weights


def flatten(the_list):
    return [i for sublist in the_list for i in sublist]


def filter_by_class(iterable, cls):
    for item in iterable:
        if isinstance(item, cls):
            yield item
