import numpy as np

from sysopt.symbolic import (
    Variable, ExpressionGraph, restriction_map, inclusion_map
)
from sysopt.backends import BackendContext

class TestNumpyAlgebra:
    # def test_numpy_algebra(self):
    #
    #     v = Variable(shape=(4,))
    #     zero = np.zeros((4, ))
    #     # result = v + zero
    #     #  assert isinstance(result, ExpressionGraph)
    #
    #     scalar_zero = zero.T @ v
    #     assert isinstance(scalar_zero, ExpressionGraph)
    #     assert (scalar_zero.nodes[1] - zero.T == 0).all()
    #     assert scalar_zero.nodes[2] is v
    #
    #     scalar_zero = v.T @ zero
    #     assert isinstance(scalar_zero, ExpressionGraph)
    #
    #     scalar_zero = (zero.T @ v).T
    #     assert isinstance(scalar_zero, ExpressionGraph)

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


class TestVectorSpaceMappings:
    def test_restriction_map(self):
        indices = [1, 2, 3]
        test_vector = np.array([2, 3, 5, 7, 11], dtype=float)
        r = restriction_map(indices, 5)
        expected_result = np.array([3, 5, 7], dtype=float)
        result = r(test_vector)
        assert isinstance(result, ExpressionGraph)
        expect_zero = result() - expected_result
        assert isinstance(expect_zero, np.ndarray)
        assert all(v == 0 for v in expect_zero.tolist())

    def test_inclusion_map(self):
        indices = dict((i, j) for i, j in enumerate([1, 2, 3]))
        i_map = inclusion_map(indices, 3, 5)
        test_vector = np.array([3, 5, 7], dtype=float)
        expected_result = np.array([0, 3, 5, 7, 0], dtype=float)
        result = i_map(test_vector)
        assert isinstance(result, ExpressionGraph)
        expect_zero = result() - expected_result
        assert isinstance(expect_zero, np.ndarray)
        assert all(v == 0 for v in expect_zero.tolist())

    def test_implicit_restriction_map(self):
        x = Variable('x', shape=(2,))

        x_0 = x[0]

        assert x_0.shape == (1,)
        assert x_0.symbols() == {x}

        x0, x1 = x
        assert x0.shape == (1,)
        assert x1.shape == (1,)
        x_f = np.array([2, 3])
        x0_f = x0(x_f)
        x1_f = x1(x_f)
        assert float(x0_f) == 2
        assert float(x1_f) == 3


class TestNumpyFunctionWrapping:

    def test_cos_of_matmal(self):
        # bug SYS-67
        x = Variable('x', shape=(3,))
        x_1 = x[1]

        expression = np.cos(x_1)

        assert expression.symbols() == {x}
        with BackendContext():
            result = expression.call({x: [0, 0, 0]})
        assert result == 1

