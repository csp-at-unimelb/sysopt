

import numpy as np
from scipy.linalg import solve_continuous_are
from codesign import *
from functools import lru_cache

x_goal = np.array([0, 0, 0,np.pi])
Q = np.diag([0.1, 1, 0.1, 1])
R = np.atleast_2d(np.array([1]))

def test_cartpole():
    def _map_params(p):
        M_c, m, l, I, b, g = p
        return [
            I + m * l * l,
            l * m,
            M_c + m,
            b,
            g
        ]


    def cart_dynamics(t, x, u, p):
        c0, c1, c2, b, g = _map_params(p)
        cx = cos(x[3])
        sx = sin(x[3])
        r = 1/(c0 * c2 - (cx * c1)**2)

        dx0 = r * (
                c0 * c1 * sx * x[1]**2
                + g * sx * cx * c1 ** 2
                + u
                - c0 * b * x[0]
        )
        dx1 = c1 * r * (
             b * cx * x[0]
             - c2 * g * sx
             - cx * (u + c1 * sx * x[1] ** 2)
        )
        dx2 = x[0]
        dx3 = x[1]

        return dx0, dx1, dx2, dx3


    class LQRController(Function):
        def __init__(self, name, goal, Q, R, *args, **kwargs):
            super(LQRController).__init__(name, *args, **kwargs)
            self.Q = Q
            self.R = R
            self.A = np.zeros_like(Q)
            n, _ = Q.shape()
            m, _ = R.shape()
            self.B = np.zeros(shape=(n, m))
            self.x_goal = goal

        @lru_cache(1)
        def _compute_care(self, p):
            c0, c1, c2, c3, b, g = p
            r = 1 / (c0 * c2 - c1 ** 2)
            (A, B) = self.A, self.B
            A[0, 0] = -c3 * c0 * r
            A[0, 3] = r * g * c1 ** 2
            A[1, 0] = -c1 * c3 * r
            A[1, 3] = c1 * c2 * g * r
            A[2, 2] = 1
            A[3, 3] = 1
            B[0, 0] = c0 * r
            B[0, 1] = c1 * r

            P = solve_continuous_are(A, B, self.Q, self.R)
            return -B.T.dot(P)

        def call(self, t, x, p):
            X = x - self.x_goal
            K = self._compute_care(p)
            u = K.dot(X)
            return u


    def cartpole_energy_shaping(x, p):
        p0, p1, p2, p3, g, k0, k1, k2 = p

        vel, ang_vel, pos, angle = x

        c3 = cos(angle)
        # m dtheta^2 /2 + mlg * (cos theta - 1)

        energy = p0 * ang_vel ** 2 - p1 * g * c3  # min at c3 -1 -> theta = pi
        # Energy Shaping Control and PD
        u = k0 * energy * c3 * ang_vel - k1 * vel - k2 * pos
        # u_d = - k1 * x[2] - k2 * x[0] - k0 * x[1]

        # feedback linearisation
        return u


    def cartpole_pfl(x, p):
        u_d = cartpole_energy_shaping(x, p)

        p0, p1, p2, p3, g, *args = p
        vel, ang_vel, _, phase = x
        c3 = cos(phase)
        s3 = sin(phase)
        u = p3 * vel - p1 * s3 * vel ** 2
        u += u_d * (p2 - (p1 * c3) ** 2 / p0) - p1 ** 2 * g * s3 * c3 / p0
        return u


    class CartPoleSwingUpController(Function):
        def __init__(self, parameters):
            super().__init__('CartPole Swingup Controller')

            self.lqr_controller = LQRController(goal=x_goal, Q=Q, R=R)
            assert len(parameters) == 8
            self.parameters = parameters

        def call(self, t, x, p):
            if abs(x[3]) < np.pi /3:
                return self.lqr_controller.call(t, x, p)
            else:
                return cartpole_pfl(x, p)


    decision_variables = [
        Variable(name='length', value=1, bounds=[0.5, 1.5]),
        Variable(name='Energy shaping gain', value=0.5, bounds=[0.01, 1])
    ]

    cart_parameters = {
            'cart mass': 1,
            'pole mass': 1,
            'pole length': decision_variables[0],
            'inertial moment': 0,
            'friction coeff': 0,
            'gravitational constant': 1
    }

    controller_params = {
        'p0': 1,
        'p1': 1,
        'p2': 3,
        'p3': 0,
        'g': 1,
        'k0': decision_variables[1],
        'k1': 0.05,
        'l2': 0.05
    }


    plant = OdeBlock(name='Cart Pole', f=cart_dynamics, u_dim=1, parameters=cart_parameters)
    controller = CartPoleSwingUpController(controller_params)

    s = System(
        name='CartPole Swingup',
        components=[plant, controller],
        wires=[(plant.output, controller.inputs),
               (controller.outputs, plant.inputs)]
    )

    x = plant.output
    u = controller.output
    loss = integrate(quad_form(Q, x) + quad_form(R, u))
    constraints = [x[2] <= 1, x[2] >= 1]
    problem = Problem(s, loss, constraints)

    soln = problem.solve()
