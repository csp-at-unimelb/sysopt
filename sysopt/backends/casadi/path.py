"""Module for differentiable path solvers."""

import casadi as _casadi
import numpy as np

from sysopt.backends.casadi.math import heaviside


class InterpolatedPath(_casadi.Callback):
    """Function class for 1d cubic interpolation between gridpoints.

    Args:
        name: Function name
        t: 1xN Array of data for the independant variable
        x: MxN array of data of M-dimensional vectors at the nth sample point.
        opt: Casadi options.

    """

    # pylint: disable=dangerous-default-value
    def __init__(self,
                 name: str,
                 t,
                 x,        # As per CasADI docs.
                 opts={}):
        super().__init__()
        self.t = _casadi.DM(t) if isinstance(t, np.ndarray) else t
        self.x = _casadi.DM(x) if isinstance(x, np.ndarray) else x

        self.construct(name, opts)

    def __len__(self):
        return self.t.shape[1]

    def __getitem__(self, item):
        return self.t[item], self.x[:, item]

    @property
    def shape(self):
        return self.x.shape

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_out(self, i):
        return _casadi.Sparsity.dense((self.x.shape[0], 1))

    def eval(self, arg):
        # return self.linearly_interpolate(arg[0])
        return self.cubic_interpolate(arg[0])

    def linearly_interpolate(self, t):

        t_lower = t - self.t[:-1]
        t_upper = t - self.t[1:]
        dt = t_upper - t_lower
        window = heaviside(t_lower) - heaviside(t_upper)
        start = heaviside(self.t[0] - t)
        end = heaviside(t - self.t[-1])

        s_lower = window * t_upper / dt
        s_upper = - window * t_lower / dt

        x_start = start * self.x[:, 0]
        x_end = end * self.x[:, -1]
        x_t = (
            self.x[:, :-1] @ s_lower.T + self.x[:, 1:] @ s_upper.T
            + x_start + x_end
        )

        return [x_t]

    def cubic_interpolate(self, t):
        # pylint: disable=invalid-name

        n = self.x.shape[0]

        t_lower = t - self.t[:-1]
        t_upper = t - self.t[1:]

        dt = t_upper - t_lower
        window = heaviside(t_lower) - heaviside(t_upper)
        start = heaviside(self.t[0] - t)
        end = heaviside(t - self.t[-1])

        s_lower = t_upper / dt
        s_upper = - t_lower / dt

        x_start = start * self.x[:, 0]
        x_end = end * self.x[:, -1]

        T = _casadi.repmat(self.t, (n, 1))

        dx = _casadi.horzcat(
            (self.x[:, 1] - self.x[:, 0]) / (T[:, 1] - T[:, 0]),
            (self.x[:, 2:] - self.x[:, :-2]) / (T[:, 2:] - T[:, :-2]) / 2,
            (self.x[:, -1] - self.x[:, -2]) / (T[:, -1] - T[:, -2])
        )

        c_0 = window * s_lower * s_lower * s_lower
        c_1 = window * s_lower * s_lower * s_upper
        c_2 = window * s_lower * s_upper * s_upper
        c_3 = window * s_upper * s_upper * s_upper

        p0 = self.x[:, :-1]
        p1 = self.x[:, :-1] + dx[:, :-1] / 3
        p2 = self.x[:, 1:] - dx[:, 1:] / 3
        p3 = self.x[:, 1:]

        x_t = (
            p0 @ c_0.T + 3 * p1 @ c_1.T + 3 * p2 @ c_2.T + p3 @ c_3.T
            + x_start + x_end
        )

        return [x_t]
