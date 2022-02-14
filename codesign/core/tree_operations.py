from typing import Tuple, List, Union

from codesign.core.scalars import Variable, Atomic
from codesign.core.vectors import Vector
from codesign.core.tree_base import ExpressionNode, Numeric, Ops
from codesign.core.differentials import diff, Derivative, VectorDerivative, is_differential


def is_integral(node: ExpressionNode):
    try:
        return node.op == 'integral'
    except AttributeError:
        return False


def has_differential(node: ExpressionNode):
    # recursive case
    try:
        return has_differential(node.lhs) or has_differential(node.rhs)
    except AttributeError:
        pass

    # base case - either atomic or somehting that we take to be static

    return is_differential(node)


_inequalities = {'<=', '>='}


def is_inequality(node: ExpressionNode):
    try:
        return node.op in _inequalities
    except AttributeError:
        return False


def integrand(node: ExpressionNode):
    assert is_integral(node), "Not an integral"
    return node.rhs


def bake_atom(atom, substitutions):

    if isinstance(atom, Numeric):
        return atom

    if isinstance(atom, Atomic) and atom in substitutions:
        return substitutions[atom]

    raise NotImplementedError("Don't know how to bake {node}")


def remove_integrals(parent, tree: ExpressionNode) -> Tuple[ExpressionNode, List[Variable], List[ExpressionNode]] :

    try:
        children = (tree.lhs, tree.rhs)
    except AttributeError:
        # base case, something other than an expression node
        return tree, [], []
    new_children = []
    new_expressions = []
    new_variables = []
    for child in children:
        if is_integral(child):
            slack = Variable()
            new_children.append(slack)
            new_variables.append(slack)
            subtree, sub_slack_vars, sub_expressions = remove_integrals(
                parent, diff(slack) == child
            )
            new_expressions.append(subtree)
        else:
            subtree, sub_slack_vars, sub_expressions = remove_integrals(
                parent, child
            )
            new_children.append(subtree)

        new_expressions += sub_expressions
        new_variables += sub_slack_vars

    return ExpressionNode(tree.op, *new_children), new_children, new_expressions


def substitute(node: ExpressionNode, mapping) -> Union[ExpressionNode, Atomic, Numeric]:
    # recursive case
    try:
        return ExpressionNode(
            node.op,
            substitute(node.lhs, mapping),
            substitute(node.rhs, mapping)
        )
    except AttributeError:
        pass

    # base case, does not have a lhs, rhs

    if isinstance(node, Vector):
        for src, dest in mapping:
            if src is node:
                return dest
        return node

    if isinstance(node, Atomic):
        for sources, targets in mapping:
            if isinstance(sources, Atomic):
                if sources is node:
                    return targets
                else:
                    continue
            if sources is not None and node in sources:
                idx = sources.index(node)
                return targets[idx]         # scalar substitution

    return node                             # none of the above


def partition_derivatives(node: ExpressionNode):
    """

    """
    # split Phi(\dot{x},x, u, p)
    # into either
    # \dot{x} = f(x, u, p)
    # or
    #      0  = g(x, u, p)
    # or
    # \dot{x} = z
    #       0 = g(x, z, u, p)

    if node.op != Ops.eq:
        raise NotImplementedError("Reduction of differential inequalities "
                                  "not yet implemented")
    lhs, rhs = node.children
    is_diff = has_differential(lhs), has_differential(rhs)

    if not any(is_diff):
        return [], [ExpressionNode(Ops.eq, lhs - rhs, 0)], []

    elif isinstance(lhs, Derivative) and not is_diff[1]:
        return [(lhs, rhs)], [], []
    elif isinstance(rhs, Derivative) and not is_diff[0]:
        return [(rhs, lhs)], [], []

    derivatives = [a for a in node.atoms() if is_differential(a)]
    new_atoms = [Variable() for _ in derivatives]
    ode_pairs = list(zip(derivatives, new_atoms))

    constraints = [substitute(node, ode_pairs)]

    return ode_pairs, constraints, new_atoms
