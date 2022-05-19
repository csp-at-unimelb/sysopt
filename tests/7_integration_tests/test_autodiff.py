
import sysopt
import numpy as np
from sysopt.blocks.builders import FullStateOutput
from sysopt.solver import SolverContext
metadata = sysopt.Metadata(states=['x'], parameters=['a', 'x0'])


def expected_result():
    dx = np.exp(-1) * np.array([-6.0, 1.0])
    dp = np.array([0.1, -2])
    return float(np.dot(dx, dp))


def dfdx(t, x, u, p):
    return -x[0] * p[0]


def x0(p):
    return p[1]


block = FullStateOutput(metadata, dfdx, x0)


def test_autodiff():

    t_f = 10
    with SolverContext(block, t_f) as soln:
        f = soln.get_integrator()
        t = 2
        p = [0.5, 3]
        dp = [0.1, -2]
        result = f.pushforward(t, p, dp)

    expected = expected_result()
    assert abs(result - expected) < 1e-6
