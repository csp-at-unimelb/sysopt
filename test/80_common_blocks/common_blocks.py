import numpy as np
from codesign.core import Signature, Parameter
from codesign.blocks.common import LinearSystem, Gain
from codesign.backends.pure_python import evaluate, lambdify


class TestGain:
    def test_signature_and_atoms(self):
        p1 = Parameter()
        p2 = Parameter()
        assert p1 is not p2

        params = [p1, p2]
        array = np.array([2, 3])
        mixed = [p1, 3, p2, 4]
        test_pairs = [
            (Gain(2), Signature(inputs=1, outputs=1)),
            (Gain(p1), Signature(inputs=1, outputs=1, parameters=1)),
            (Gain(array), Signature(inputs=2, outputs=2)),
            (Gain(params), Signature(inputs=2, outputs=2, parameters=2)),
            (Gain(mixed), Signature(inputs=4, outputs=4, parameters=2))
        ]

        for i, (block, signature) in enumerate(test_pairs):
            assert block.signature == signature, \
                f"Failed on case {i}: {block.gain}"
            assert len(block.parameters.atoms()) == signature.parameters, \
                f"Failed on case {i}: parameters and atoms misaligned"

    def test_expressions(self):
        p1 = Parameter()
        p2 = Parameter()
        params = [p1, p2, 2]
        block = Gain(params)
        expressions = block.expressions()

        assert len(expressions) == 3

        inputs = [1, 1/2, 1/3]
        parameters = [3, 5]
        outputs = [3, 5/2, 2/3]

        arguments = list(block.inputs) + list(block.parameters) + list(block.outputs)
        correct_args = inputs + parameters + outputs
        wrong_args = inputs + parameters + [0, 0, 0]
        for expression in expressions:
            assert expression.op == '=='
            residue = lambdify(expression, *arguments)
            result = residue(*correct_args)
            assert result == 0
            assert residue(*wrong_args) != 0
