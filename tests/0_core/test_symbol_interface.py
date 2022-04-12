import numpy as np

from sysopt.symbolic import Variable, ExpressionGraph


class TestNumpyAlgebra:
    def test_numpy_algebra(self):

        v = Variable(shape=(4,))
        zero = np.zeros((4, ))
        result = v + zero
        assert isinstance(result, ExpressionGraph)

        scalar_zero = zero.T @ v
        assert isinstance(scalar_zero, ExpressionGraph)
        assert (scalar_zero.nodes[1] - zero.T == 0).all()
        assert scalar_zero.nodes[2] is v

        scalar_zero = v.T @ zero
        assert isinstance(scalar_zero, ExpressionGraph)

        scalar_zero = (zero.T @ v).T
        assert isinstance(scalar_zero, ExpressionGraph)

    def test_slicing(self):
        v = Variable(shape=(4, ))
        v_slice = v[0: 2]
        assert isinstance(v_slice, ExpressionGraph)
        assert isinstance(v_slice.T, ExpressionGraph)

        vt = v[0:2].T
        assert vt.shape == (1, 2)
        assert vt is not v_slice

        vtt = vt.T
        assert vtt != vt
        assert vt.shape != vtt.shape
        product = vt @ vtt
        assert product.shape == (1, 1)
