import numpy as np
from codesign.core import Signature, Parameter
from codesign.blocks.common import LinearSystem
from codesign.backends.pure_python import lambdify


def test_api():
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[1], [0]])
    x0 = np.array([0, 0])
    lti = LinearSystem(A=A, B=B, x0=x0)

    assert lti.signature.inputs == 1
    assert lti.signature.outputs == 2
    assert lti.signature.state == 2
    assert lti.signature.parameters == 0

    f,g,h = lti.expressions()