import numbers

import numpy as np

from codesign import ExpressionNode, Atomic, Vector, DenseArray, Ops


def evaluate_atom(atom: Atomic, subs):
    try:
        return float(atom)
    except (TypeError, ValueError):
        pass

    for key, value in subs:
        if key is atom:
            return value
        elif isinstance(key, Vector):
            for inner_key, inner_value in zip(key, value):
                if inner_key is atom:
                    return inner_value

    raise ValueError(f"Dont know how to convert {atom} to a numerical")


def evaluate(self: ExpressionNode, subs):
    # recursive case
    try:
        lhs, rhs = evaluate(self.lhs, subs), evaluate(self.rhs, subs)
        return apply(self.op, lhs, rhs)
    except AttributeError:
        pass

    # base case
    if isinstance(self, numbers.Number):
        return self

    if isinstance(self, (Vector, DenseArray)):
        return np.array([evaluate_atom(d, subs) for d in self.data])\
            .reshape(self.shape)

    return evaluate_atom(self, subs)


def apply(op, lhs, rhs):
    if op == Ops.add:
        return lhs + rhs
    elif op == '-':
        return lhs - rhs
    elif op == Ops.mul:
        return lhs * rhs
    elif op == Ops.matmul:
        return lhs @ rhs
    elif op == Ops.power:
        return pow(lhs, rhs)

    raise NotImplementedError(f'Operation {op} not implemented')


class lambdify:
    def __init__(self, expr, *arguments: Atomic):

        assert not set(expr.atoms()) - set(arguments), \
            "Result has unevaluated symbols"

        self._expr = expr
        self._args = arguments

    def __call__(self, *args):
        if len(args) != len(self._args):
            raise TypeError(
                f"Invalid arguments. Wanted {len(self._args)},"
                f" but got {len(args)} "
            )
        for arg in args:
            if not isinstance(arg, numbers.Number):
                raise TypeError(f"Cannot evaluate expression with type {type(arg)}")

        substitutions = list(zip(self._args, args))

        if self._expr.op == Ops.eq:
            return evaluate(self._expr.lhs - self._expr.rhs, substitutions)
        elif self._expr.op == Ops.leq:
            # a <= b -> 0 <= b - a
            # so return f = max(a - b, 0) so that f == 0 implies a < b
            return max(
                evaluate(self._expr.lhs - self._expr.rhs, substitutions), 0
            )

        return evaluate(self._expr, substitutions)
