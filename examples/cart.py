import termios

import matplotlib.pyplot as plt
import numpy as np
import scikits.odes as sko
# from codesign.direct import direct
# Cart function

from matplotlib.widgets import Rectangle, Line2D, Circle
from matplotlib.animation import TimedAnimation


b = 0.1
g = 1

# c_0 = 1 -> M_c =0, m = 1, l = 1, I =0


def _map_params(p):
    M_c, m, l, I = p
    return [
        I + m * l * l,
        l * m,
        M_c + m
    ]


def plant_dynamics(t, x, dx, u, p):
    c0, c1, c2 = _map_params(p)
    cos = np.cos(x[3])
    sin = np.sin(x[3])
    r = 1/(c0 * c2 - (cos * c1)**2)

    dx[0] = r * (
            c0 * c1 * sin * x[1]**2
            + g * sin * cos * c1 ** 2
            + u
            - c0 * b * x[0]
    )
    dx[1] = c1 * r * (
         b * cos * x[0]
         - c2 * g * sin
         - cos * (u + c1 * sin * x[1] ** 2)
    )
    dx[2] = x[0]
    dx[3] = x[1]


def plant_dynamics_old(t, x, dx, u, p):
    p0, p1, p2 = _map_params(p)

    M = np.eye(4, dtype=float)

    G = np.array([1, 0, 0, 0], dtype=float)
    c3 = np.cos(x[3])
    s3 = np.sin(x[3])
    M[0, :] = [p2, p1 * c3, b, 0]
    M[1, :] = [p1 * c3, p0, 0, 0]
    F = np.array([p1 * s3 * x[1]**2, -g * p1 * s3, x[0], x[1]])
    M_inv = np.linalg.inv(M)

    dx[0:4] = M_inv @ (F + G * u)


from scipy.linalg import solve_continuous_are

x_goal = np.array([0, 0, 0, np.pi], dtype=float)
Q = np.diag([0.1, 1, 0.1, 1])


def lqr_stabiliser(x, p):
    c0, c1, c2, c3, k0, k1, k2 = p
    X = x[0:4] - x_goal
    r = 1 / (c0 * c2 - c1 ** 2)
    A = np.array(
        [[-c3 * c0 * r, 0, 0, r * g * c1**2],
         [-c1*c3 * r, 0, 0, c1 * c2 * g * r],
         [1, 0, 0, 0],
         [0, 1, 0, 0]
    ])

    B = np.array([[c0 * r], [c1 * r], [0], [0]])

    R = np.array([[1]], dtype=float)

    P = solve_continuous_are(A, B, Q, R)
    u = - B.T.dot(P).dot(X)
    return u


def pfl_shaping_controller(x, p):
    p0, p1, p2, p3, k0, k1, k2 = p
    c3 = np.cos(x[3])
    s3 = np.sin(x[3])
    # m dtheta^2 /2 + mlg * (cos theta - 1)
    energy = p0 * x[1]**2 - p1 * g * c3  # min at c3 -1 -> theta = pi

    # Energy Shaping Control and PD
    u_d = k0 * energy * c3 * x[1] - k1 * x[2] - k2 * x[0]
    # u_d = - k1 * x[2] - k2 * x[0] - k0 * x[1]

    # feedback linearisation
    u = p3 * x[0] - p1 * s3 * x[1]**2 + u_d * (p2 - (p1 * c3)**2 / p0) - p1**2 * g * s3 * c3/p0

    return u


def controller(x, p):

    if abs(x[3] - np.pi) < np.pi / 3:
        return lqr_stabiliser(x, p)

    return pfl_shaping_controller(x, p)


def clamp(x):
    x[3] = np.fmod(x[3], 2 * np.pi)
    return x


def one_step_loss(x, u):
    error = clamp(x[0:4] - x_goal)
    return error.dot(Q.dot(error)) + 0.1 * u ** 2


# MPC controller
def mpc_controller(x, p):
    # minimise x_4[K] -> loss
    # such that
    # x[k + 1] = x[k] + F(x[k]) + G(x[k]) u[k]
    # or  M(x[k]) (x[k+ 1] - x[k]) - dt * F(X[k]) = dt * G * u[k + 1]

    # loss = (x - x_g).T Q (x - x_g) + u.T R u
    # st P_i(x, u) = 0 -> (3rd order polynomial)
    # for all i
    pass


def closed_loop_dynamics(t, x, dx, p):
    p_controller, p_plant = p
    u = controller(x, p_controller)
    plant_dynamics(t, x, dx, u, p_plant)
    dx[4] = one_step_loss(x, u)


class DesignParameter:
    def __init__(self, n, defaults=None):
        self._n = n
        if n > 1:
            self._value = np.zeros(shape=(n,))
        else:
            self._value = 0

    @property
    def value(self):
        return self._value

    @property
    def bounds(self):
        return [-np.inf, np.inf]


plant_params = DesignParameter(4)
# mass, mass, length, moment
plant_params._value[:] = [1, 1, 1, 0]
controller_params = DesignParameter(7)
controller_params._value[:] = list(_map_params(plant_params.value)) + [b, 1, 0.05, 0.05]


class InvalidParameterException(Exception):
    pass


def _bake_param(p):
    if isinstance(p, DesignParameter):
        return p.value
    if isinstance(p, (float, int, complex)):
        return p
    if isinstance(p, np.ndarray):
        return p

    raise InvalidParameterException(f"Don't know what to do with parameter {p}")


def solve(params, loss_only=False):
    values = [_bake_param(p) for p in params]
    T_final = 100

    def rhs(t, y, ydot):
        closed_loop_dynamics(t, y, ydot, values)

    solver = sko.ode('CVODE', rhs)
    initial_values = np.zeros((5,), dtype=float)
    initial_values[3] = 0.2
    t_out = np.linspace(0, T_final, 1000)

    output = solver.solve(t_out, initial_values)

    if loss_only:
        return output.values.y[4, -1]
    else:
        return t_out, output.values.y


def search():
    from codesign.direct import direct
    params = (controller_params, plant_params)

    def func(x, loss_only=True):
        #params[1]._value[3] = x[0]
        # params[0]._value[0:4] = _map_params(params[1].value)
        params[0]._value[-3:] = x
        t, path = solve(params)
        x_final = path[:, -1]
        loss = x_final[4] + t.max() * one_step_loss(x_final, 0)
        if loss_only:
            return loss
        else:
            return t, path, loss

    sol, history, _ = direct(func, [(1, 2), (0, 1), (0, 1)], steps=150)

    t, Y, _ = func(sol.argmin, loss_only=False)
    print(f"Search conveged to {sol.min} at {sol.argmin} after {sol.calls}")
    print(f"Final position: {Y[-1, 0:4]}")
    import matplotlib.pyplot as plt

    plt.plot(Y[:, 3], Y[:, 1])
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.xlabel('Angle from -z')
    plt.ylabel('Angular Velocity')
    plt.show()

    params[0]._value[-3:] = sol.argmin[1:]

    plot_control_effort(t, Y, params[0].value)
    plot_optimisation_path(history)


def plot_optimisation_path(rects):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    v, p1, p2, p3 = zip(*((r.value, r.center[0], r.center[1], r.center[2]) for r in rects))

    ax1.set_title('Best Estimate at each step')
    ax1.plot(p1, label='p_1')
    ax1.plot(p2, label='p_2')
    ax1.plot(p3, label='p_3')
    ax2.semilogy(v, label='Loss')
    ax2.set_xlabel('Step')
    plt.show()


class CartPoleAnimation(TimedAnimation):
    def __init__(self):
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(111)
        self.params = (controller_params, plant_params)
        self.params[0]._values[-3:] = [1.1, 0.01, 0.01]
        self.cart = plt.Rectangle(xy=(-0.25, -0.125), width=0.5, height=0.25, fill=None)
        self.pole = plt.Line2D([0, 0], [0, -1])
        self.axis.add_artist(self.cart)
        self.axis.add_artist(self.pole)
        self.axis.set_ylim(-2, 3)
        TimedAnimation.__init__(self, fig=self.figure, interval=50, blit=True)
        self.delta_T = 0.1
        self.X = np.zeros((5,), dtype=float)
        self.X[3] = 1

    def _draw_frame(self, _):
        t_out = np.linspace(0, self.delta_T, 100)
        values = [_bake_param(p) for p in self.params]

        l = self.params[1].value[2]

        def rhs(t, y, ydot):
            closed_loop_dynamics(t, y, ydot, values)

        solver = sko.ode('CVODE', rhs)
        output = solver.solve(t_out, self.X)
        self.X[:] = output.values.y[-1, :]
        x, theta = self.X[2:4]
        self.cart.set_xy((x-0.25, -0.125))
        self.pole.set_xdata([x, x + l * np.sin(theta)])
        self.pole.set_ydata([0, -l * np.cos(theta)])
        self.axis.set_xlim(x - 3, x + 3)

        return self.cart, self.pole

    def new_frame_seq(self):
        return iter(range(100))


def animate():
    anim = CartPoleAnimation()

    plt.show()

import matplotlib.pyplot as plt


def main():
    test_params = (controller_params, plant_params)
    t, Y = solve(test_params)

    plt.plot(Y[:, 3], Y[:, 1])
    plt.plot(Y[-1, 3], Y[-1, 1], 'x')
    plt.xlim([- 2*np.pi, 2 * np.pi])
    plt.ylim([-2 * np.pi, 2 * np.pi])
    plt.xlabel('Angle from -z')
    plt.ylabel('Angular Velocity')
    plt.show()

    plot_control_effort(t, Y, controller_params.value)


def sweep3d():
    n = 25
    params = (controller_params, plant_params)
    L, K0 = np.meshgrid(
        np.linspace(0.5, 1.5, n),
        np.linspace(1, 2, n)
    )

    def func(x, loss_only=True):
        params[1].value[3] = x[0]
        params[0]._value[-3] = x[1]
        t, path = solve(params)
        x_final = path[:, -1]
        loss = one_step_loss(x_final, 0)
        if loss_only:
            return loss
        else:
            return t, path, loss

    LOSS = np.empty_like(L)
    for i in range(n):
        for j in range(n):
            LOSS[i, j] = func((L[i, j], K0[i, j]))
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(L, K0, LOSS, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Pole Length')
    ax.set_ylabel('ES. Gain')
    ax.set_zlabel('Terminal Error')
    plt.show()

def sweep2d():
    n = 150
    params = (controller_params, plant_params)
    L = np.linspace(0.5, 1.5, n)

    def func(x, loss_only=True):
        params[1].value[3] = x[0]
        t, path = solve(params)
        x_final = path[:, -1]
        loss = one_step_loss(x_final, 0)
        if loss_only:
            return loss
        else:
            return t, path, loss

    LOSS = np.empty_like(L)
    for i in range(n):
            LOSS[i] = func(L)

    plt.plot(L, LOSS)
    ax = plt.gca()
    ax.set_xlabel('Pole Length')
    ax.set_ylabel('Terminal Error')
    plt.show()

def plot_control_effort(t, x, p):

    u = np.empty_like(t)
    for i in range(len(u)):
        u[i] = controller(x[i,:], p)
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True)
    ax1.plot(t, x[:, 3], label='Angle')
    ax1.plot(t, x[:, 1], '--', label='Ang. Vel')
    ax1.legend()
    ax2.plot(t, x[:, 0], '--', label='Cart Vel.')
    ax2.plot(t, x[:, 2], label='Cart Pos.')
    ax2.legend()
    ax3.plot(t, u, label='Force')
    ax3.set_xlabel('t')
    ax3.legend()
    ax1.set_title('Optimized solution')
    plt.show()


if __name__ == '__main__':
    # animate()
    # main()
    # search()
    sweep2d()
