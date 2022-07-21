import numpy as np
from sysopt.symbolic import Function, symbolic_vector


def f(x, y):
    return x + y, x**2 + y**2


def dfdx(x, y):
    return (1, 1), (2 * x, 2*y)

def dF(x, y, dx, dy):
    return dx + dy, 2 * (x +y)

def test_function_wrapper_numeric():
    F = Function(
        function=f,
        jacobian=dfdx,
        shape=(2,),
        arguments=[symbolic_vector('x'), symbolic_vector('y')]
    )

    three, five = F(1, 2)
    assert three == 3
    assert five == 5

    two_F = 2 * F

    six, ten = two_F(1, 2)
    assert six == 6
    assert ten == 10


def test_function_wrapper_closures():
    arguments = [symbolic_vector('x'), symbolic_vector('y')]
    F = Function(
        function=f,
        jacobian=dfdx,
        shape=(2,),
        arguments=arguments
    )
    x = symbolic_vector('x')
    assert F.shape == (2,)
    Fx = F(x, 2)
    assert Fx.shape == F.shape
    expected_result = (3, 5)
    assert Fx.symbols() == {x}

    result = Fx.call({x: 1})
    assert result == expected_result

    X = symbolic_vector('X', 2)
    FX = F(X[0], X[1])
    assert FX.symbols() == {X}

    Xn = np.array([1, 2])
    assert expected_result == FX(Xn)

    result = FX.call({X: Xn})
    assert result == expected_result

    expected_result = (6, 10)
    two_Fx = 2 * FX
    six, ten = two_Fx(Xn)
    assert (six, ten) == expected_result

    six, ten = two_Fx.call({X: Xn})
    assert (six, ten) == expected_result


def test_forwards_derivatives():
    arguments = [symbolic_vector('x'), symbolic_vector('y')]
    F = Function(
        function=f,
        forwards=dF,
        shape=(2,),
        arguments=arguments
    )

    assert F.forwards is not None
    assert F.forwards(0, 1, 2, 3) == dF(0,1,2,3)
