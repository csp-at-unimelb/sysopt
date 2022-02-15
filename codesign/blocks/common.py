import numpy as np

from codesign import Block, Signature, Vector, Parameter, diff
from numpy import eye


class LinearSystem(Block):
    def __init__(self, A, B, x0, C=None):
        self.A = A
        self.B = B
        n_x, m_x = A.shape
        n_u, m_u = B.shape

        self.C = C if C is not None else eye(n_x)
        n_y, m_y = C.shape

        assert n_x == len(x0), "A matrix not same size as Initial coniditions"
        assert n_x == m_x,  "A matrix is not square"
        assert n_u == n_x,  "B row space is not the same as A row space"
        assert m_y == n_x,  "C row space is not the same as A row space"
        m_u = max(m_u, 1)   # take care of the edge case where b is a vector
        self.x0 = x0

        params = set(A.atoms()) | set(B.atoms())
        try:
            params |= set(x0.atoms())
        except AttributeError:
            pass

        super().__init__(
            signature=Signature(
                inputs=m_u,
                outputs=n_y,
                state=n_x,
                parameters=len(params)
                ),
            parameters=params
        )

    def expressions(self):
        x = self.state
        u = self.inputs
        return (self.A @ x + self.B @ u,
                self.C @ x, [])


class Gain(Block):
    def __init__(self, gain):

        if isinstance(gain, (list, tuple, np.ndarray)):
            self.gain = Vector(gain)
        else:
            self.gain = Vector([gain])

        channels = len(self.gain)
        atoms = set(self.gain.atoms())
        params = {p for p in atoms if isinstance(p, Parameter)}

        assert not atoms - params, "Gain must be a constant!"

        sig = Signature(inputs=channels,
                        outputs=channels,
                        state=0,
                        parameters=len(params))

        super().__init__(signature=sig, parameters=params)

    def expressions(self):
        return ([],
                [self.inputs * self.gain],
                [])

