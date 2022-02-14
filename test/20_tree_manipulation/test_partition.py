from codesign import ExpressionNode, Variable, Parameter, diff, Ops
from codesign.core.tree_operations import partition_derivatives


def cmp_op(op1, op2):
    return (op1.op == op2.op) and op1.lhs is op2.lhs and op1.rhs is op2.rhs


def test_partition_ode():
    p = Parameter()
    x = Variable()
    y = Variable()
    dx = diff(x)

    expression = dx == y + p

    assert isinstance(expression, ExpressionNode)

    ode_pairs, constraints, new_derivatives = partition_derivatives(expression)

    assert ode_pairs[0][0] is dx
    assert ode_pairs[0][1].op == Ops.add
    assert ode_pairs[0][1].lhs is y

    assert not constraints
    assert not new_derivatives


def test_partition_constraint():

    x = Variable()
    y = Variable()

    expression = x == y
    assert isinstance(expression, ExpressionNode)

    ode_pairs, constraints, new_atoms = partition_derivatives(expression)

    assert not ode_pairs
    assert not new_atoms
    constraint, = constraints
    assert constraint.rhs == 0
    assert constraint.op == '=='
    assert constraint.lhs.op == '-'
    assert constraint.lhs.lhs is y
    assert constraint.lhs.rhs is x


def test_partition_dae():
    p = Parameter()
    x = Variable()
    y = Variable()
    dx = diff(x)

    expression = dx ** 2 + y ** 2 == p
    ode_pairs, constraints, new_atoms = partition_derivatives(expression)

    assert ode_pairs
    assert constraints
    assert new_atoms

    z, = new_atoms

    assert ode_pairs[0][0] is dx
    assert ode_pairs[0][1] is z

    # todo: FIXME (requires comparing trees)
    # assert constraints is z ** 2 + y ** 2 - p == 0
