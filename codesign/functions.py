from codesign.core import ExpressionNode, t, Ops
from codesign.core import Derivative, Variable
from numpy import sin, cos


def dot(x, y):
    return ExpressionNode(Ops.dot, x, y)


def sum_of_squares(x):
    return dot(x, x)


def quad_form(Q, x):
    return dot(x, Q @ x)


def time_integral(x: ExpressionNode):

    return ExpressionNode(Ops.integral, t, x)

