from codesign import Variable, Parameter, ExpressionNode


def test_addition():

    x = Variable()
    p = Parameter()

    expr = x + p
    equal_expr = expr == 0

    assert isinstance(expr, ExpressionNode)

    assert isinstance(equal_expr, ExpressionNode)


def test_subtraction():
    x = Variable()
    p = Parameter()

    expr = x - p
    assert isinstance(expr, ExpressionNode)

    assert isinstance(expr == 0, ExpressionNode)

