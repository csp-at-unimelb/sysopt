import pytest
from codesign import ExpressionNode, Parameter, Variable

pytest.mark.skip()
def test_equality():
    p1 = Parameter()
    p2 = Parameter()

    expr = p1 + p2 == 0
    expr2 = p1 + p2 == 0
    expr3 = 0 == p1 + p2
    expr4 = 1 == p1 + p2




