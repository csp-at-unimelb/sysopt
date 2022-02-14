from codesign import (
    Variable, Vector, diff, is_differential
)
from codesign.core.differentials import is_scalar


def test_derivative_creating_and_querying():

    x = Variable()
    dx = diff(x)
    assert not is_differential(x)
    assert is_differential(dx)
    assert dx.parent is x

    X = Vector.create_filled_with(Variable, 3)
    DX = diff(X)

    assert not is_differential(X)
    assert is_differential(DX)
    assert DX.parent is X
    assert X.shape == DX.shape

    for DX_i, X_i in zip(DX, X):
        assert DX_i.parent is X_i
