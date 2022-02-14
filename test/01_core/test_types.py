from codesign import Parameter, Variable


def test_parameters():
    p1 = Parameter()
    p2 = Parameter()

    assert p1 is not p2

    assert hash(p1) == id(p1)
    assert p1 in {p1, p2}


def test_variable():
    x1 = Variable()
    x2 = Variable()
    x3 = x1
    assert x1 is not x2
    assert hash(x1) == id(x1)

    assert x1 in {x1, x2}
    assert {x1, x3} == {x1}



