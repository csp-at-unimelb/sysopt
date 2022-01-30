import numpy as np
from dataclasses import dataclass


@dataclass
class HyperRect:
    center: np.ndarray
    sides: np.ndarray
    value: float

    @property
    def size(self):
        return np.linalg.norm(self.sides / 2)


def basis_vectors(bool_array):
    for i, keep in enumerate(bool_array):
        if keep:
            v = np.zeros_like(bool_array, dtype=float)
            v[i] = 1
            yield v


def split_rect(func, rect: HyperRect):

    max_length = rect.sides.max()
    evals = []
    delta = max_length / 3
    for e_k in basis_vectors(rect.sides == max_length):
        c_plus = rect.center + delta * e_k
        c_minus = rect.center - delta * e_k
        f_plus = func(c_plus)
        f_minus = func(c_minus)
        evals.append((e_k, c_plus, c_minus, f_plus, f_minus))

    evals.sort(key=lambda x: min(x[3], x[4]), reverse=True)

    center_rect = rect
    outputs = []
    while evals:
        e_k, c_p, c_m, f_p, f_m = evals.pop()
        outputs.append(HyperRect(center=c_p, sides=center_rect.sides - 2 * delta * e_k, value=f_p))
        outputs.append(HyperRect(center=c_m, sides=center_rect.sides - 2 * delta * e_k, value=f_m))
        center_rect.sides -= 2 * delta * e_k
    outputs.append(center_rect)
    return outputs


def lower_convex_hull(rects, f_0, norm=None):
    x0, y0 = (0, f_0)
    K = np.inf
    if norm is None:
        norm = lambda x: np.linalg.norm(x)

    P = []
    while True:
        test_rect = None
        test_i = None
        for i, r in enumerate(rects):

            x1 = norm(r)
            if x0 > x1:
                continue

            dy = r.value - y0
            dx = x1 - x0
            if 0 < dy < K * dx:
                test_rect = r
                test_i = i
                K = dy / dx

        if test_rect is None:
            return P

        x0, y0 = norm(test_rect), test_rect.value
        P.append(test_i)


@dataclass
class Solution:
    min: float
    argmin: np.ndarray
    steps: int
    calls: int
    tolerance: float


def direct(function, bounds, steps=100, eps=1e-4, use_inf_norm=False):
    offset = np.array([a for a, _ in bounds])
    scale = np.array([b - a for a, b in bounds])
    c = np.ones_like(offset, dtype=float)/2
    f = lambda x: function(offset + x * scale)

    rects = [HyperRect(center=c, sides=np.ones_like(c), value=f(c))]
    f_min = rects[0]

    history = [f_min]
    m = 1
    if use_inf_norm:
        norm = lambda x: x.sides.max() / 2
    else:
        norm = lambda x: x.size

    for t in range(steps):
        f_0 = f_min.value - eps * abs(f_min.value)

        hull_indicies = lower_convex_hull(rects, f_0, norm)
        new_rects = []
        for i in hull_indicies:
            r = rects.pop(i)
            new_rects += split_rect(f, r)
        m += len(new_rects)
        f_min = min([f_min] + new_rects, key=lambda x: x.value)
        rects += new_rects
        history.append(f_min)

    sol = Solution(
        min=f_min.value,
        argmin=(f_min.center * scale + offset),
        steps=steps,
        calls=m,
        tolerance=np.linalg.norm(f_min.sides - f_min.center)
    )
    return sol, history, [offset + r.center * scale for r in rects]



def quadratic(x):
    return 0.5*((0.5 - x[0])**2 + (0.25 - x[1])**2)


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock2(x):
    return 0.25 * abs(x[0] - 1) + abs(x[1] - 2 * abs(x[0]) + 1)


def linear(x):
    return 1 + x[0] + x[1]


def main():
    bounds = [(-2, 2), (-2, 2)]
    # bounds = [(0, 1), (0, 1)]
    f = rosenbrock2
    sol, path, c = direct(f, bounds, steps=500, use_inf_norm=True)

    import matplotlib.pyplot as plt

    plt.plot([p.value for p in path])
    plt.xlabel("Iterations")
    plt.ylabel(r"$\min_x\ f(x)$")
    plt.title("")
    plt.show()
    print(f"Converged to f({sol.argmin}) = {sol.min} after {sol.calls} evaluations")

    plt.figure()
    X, Y = np.meshgrid(np.linspace(*bounds[0]), np.linspace(*bounds[1]))
    Z = np.empty_like(Y)
    n,m = Z.shape
    for i in range(n):
        for j in range(m):
            Z[i, j] = f((X[i, j], Y[i, j]))
    plt.scatter([ci[0] for ci in c], [ci[1] for ci in c])
    z0 = Z.min()
    z_max = Z.max()

    plt.contour(X, Y, Z, np.geomspace(z0, z_max, 15), alpha=0.5)
    plt.title('Samples Taken')

    plt.show()


if __name__ == '__main__':
    main()
