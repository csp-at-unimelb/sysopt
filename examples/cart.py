from codesign.interfaces import Model, DesignParameter
import numpy as np
import cvxpy

p_default = (0.5, 0.2, 0.1, 0.3, 0.006, 9.8)
theta_range = [np.pi/2, 3 * np.pi/2]

g = 9.8

initial_conditions = [0, 0, 0, np.pi]


class CartPlant(Model):

    def variables(self):
        return [
            ['Cart Velocity', 'Cart Position', 'Angular Velocity', 'Angle from Vertical'],
            ['Force'],
            ['Cart Velocity', 'Cart Position', 'Angular Velocity', 'Angle from Vertical'],
            ['Cart Mass', 'Pendulum Mass', 'Friction Coefficient',
             'Pendulum Length', 'Pendulum mass moment of inertia']
        ]

    def initial_state(self):
        return [0, 0, 0, np.pi]

    def get_residual(self, dt, dX, t, X, u, y, p):
        M, m, b, l, I = p

        out = [
            dX[1]/dt - X[0],
            dX[3]/dt - X[2],
            (M + m) * dX[0]/dt + b * X[0] + m * l * (dX[2]/dt * np.cos(X[2]) - np.sin(X[3]) * dX[3] ** 2) - u[0],
            (I + m * l ** 2) * dX[2]/dt + m * g * l * np.sin(X[3]) + m * l * dX[0]/dt * np.cos(X[3]),
            y[0] - X[0],
            y[1] - X[1],
            y[2] - X[2],
            y[3] - X[3],
        ]
        return out

    def get_system_constraints(self, t, x, u, y, p):
        dx, x, dtheta, theta = x
        u0, = u
        out = [
            dx < 0.5,
            dx > -0.5,
            theta - np.pi < 0.35,
            theta - np.pi > -0.35
        ]
        return out

    def get_parameter_constraints(self, p):
        M, m, b, l, I = p

        return [
            l > 0.1,
            l < 0.5
        ]


class CartController(Model):
    def __init__(self):
        self.Q = np.eye(2)
        self.B = np.array([1, 0, 0, 0], dtype=float, shape=(4, 1))
        self.C = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
        self.Gx = None
        self.U = None
        self.T = None
        self.f = None
        self.problem = None
        self.max_force = 1
        self.r = 0.5

    def _model(self, horizon, A, B):
        # X[k + 1] = Ax[k] + Bu[k]
        # Y[K + 1] = C X[k]
        C = self.C

        # B = [dt, 0, 0, 0]
        n, _ = A.shape
        G = np.empty((n * horizon, A.shape[1]), dtype=float)
        G[0: n, :] = A
        for i in range(1, horizon):
            G[n * i: n*(i + 1), :] = G[n * (i - 1): n * i, :] @ A

        n_b, m_b = B.shape
        n_c, m_c = C.shape
        F = np.empty((horizon * n_c, m_b))
        H = np.zeros_like((horizon * n_c, horizon * m_b), dtype=float)
        F[0: n_b, :] = C @ B

        b = A @ B
        F[n_b: 2 * n_b, :] = C @ b
        for i in range(2, horizon):
            b = A @ b
            F[i * n_b: (i + 1) * n_b, :] = C @ b

        for i in range(1, horizon):
            H[i * n_b:, i * m_b:(i + 1) * m_b] = F[(i - 1) * n_b: - i * n_b, :]

        return G, H, F

    def variables(self):
        return [
            ['Last Control Value'],                                     # No dynamic state variables
            ['Cart Velocity', 'Cart Position', 'Angular Velocity', 'Angle from Vertical'],
            ['Force'],                               #
            ['Horizon', ('A', 4, 4)]
        ]

    def initial_state(self):
        return [0]

    def get_residual(self, dt, u_next, t, u_last, state, out, p):

        T, A = p
        horizon = int(T/dt)

        u = self.solve_qp(state, u_last[0], A, self.B*dt, horizon)

        return [
            u_next[0] - u_last - u,
            out[0] - u,
        ]

    def solve_qp(self, x, u, A, B, T):
        if self.T != T:
            self.rebuild_qp(T)

        G, H, F = self._model(T, A, B)
        self.f = H.transpose() @ ((G @ x + F @ u) + np.pi)
        self.Q = self.Q_u + H.transpose() @ H
        self.problem.solve()
        return self.U.value[0]

    def rebuild_qp(self, T):
        self.T = T
        self.U = cvxpy.Variable(T)
        D = np.eye(T) - np.diag([1]* (T - 1), 1)
        D[-1, -1] = 0
        self.Q_u = self.r * D.T @ D
        self.Q = cvxpy.Parameter(shape=(T, T))
        self.f = cvxpy.Parameter(shape=(T,))

        constraints = [u <= self.max_force for u in self.U]
        constraints += [-u <= self.max_force for u in self.U]

        objective = cvxpy.QuadForm(self.U, self.Q) + self.f.T @ self.U

        self.problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def get_parameter_constraints(self, p):
        horizon, _ = p
        return [horizon > 0]



# (T, A, \omega) -> x(t)