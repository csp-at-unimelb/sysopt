import numpy as np

from sysopt.backends.casadi import InterpolatedPath


def test_interpolated_path_ongrid():
    points = 25
    t_max = 10
    t_test = np.linspace(0, t_max, points)
    t = np.reshape(t_test, (1, 25))

    def f(_t):
        return _t**3 + 2 * _t**2 + 1

    x = f(t)
    x = np.reshape(x, newshape=(1, 25))
    path = InterpolatedPath('cubic', t, x)

    for i in range(points):
        t_i = t[0, i]
        y_i = path([t_i])

        assert abs(f(t_i) - y_i) < 1e-4


def test_interpolated_path_offgrid():
    points = 125
    t_max = 10
    t_test = np.linspace(0, t_max, points)
    t = np.reshape(t_test, (1, points))

    def f(_t):
        return _t**3 + 2 * _t**2 + 1

    x = f(t)
    x = np.reshape(x, newshape=(1, points))
    path = InterpolatedPath('cubic', t, x)

    test_interval_interior = (float(path.t[1]), float(path.t[2]))

    s = np.linspace(*test_interval_interior, 10).tolist()
    delta_s = test_interval_interior[1] - test_interval_interior[0]
    p0 = f(test_interval_interior[0])
    p3 = f(test_interval_interior[1])
    m1 = (path.x[0, 2] - path.x[0, 0]) / 2
    m2 = (path.x[0, 3] - path.x[0, 1]) / 2
    p1 = p0 + m1 / 3
    p2 = p3 - m2 / 3
    for s_i in s:
        tau = (s_i - test_interval_interior[0]) / delta_s
        h00 = 2 * tau**3 - 3*tau**2 + 1
        h10 = tau**3 - 2*tau**2 + tau
        h01 = -2 * tau**3 + 3*tau**2
        h11 = tau**3 - tau**2
        p_expected = h00 * p1 + h10*m1 + h01*p2 + h11*m2

        assert abs(f(s_i) - p_expected) < 0.1
        p_actual = path(s_i)

        assert abs(p_actual[0] - p_expected) < 0.1
