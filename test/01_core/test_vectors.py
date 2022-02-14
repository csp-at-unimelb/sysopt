from codesign import Vector, Parameter, Variable, ExpressionNode


class TestVector:
    def test_comparison(self):
        p = [Parameter()] * 2
        v = Vector(p)
        v4 = v
        assert v4 is v

    def test_identity(self):
        p = [Parameter()] * 2
        v = Vector(p)
        v2 = Vector(p)
        v3 = Vector([1, 2])
        assert len(v) == len(v2) == len(v3)
        expr_true = v == v2
        expr_false = v3 == v

        assert isinstance(expr_true, ExpressionNode)
        assert isinstance(expr_false, ExpressionNode)

    def test_atoms_interface(self):
        p1, p2 = Parameter(), Parameter()

        assert not set(Vector([1]).atoms())

        scalar_atom = set(Vector([p1]).atoms())
        single_atom = set(Vector([p1, 2]).atoms())
        double_atom = set(Vector([p1, p2]).atoms())

        assert len(scalar_atom) == 1
        assert len(single_atom) == 1
        assert len(double_atom) == 2



