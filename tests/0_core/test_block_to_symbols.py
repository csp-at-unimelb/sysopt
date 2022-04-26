from sysopt.types import *
from sysopt.block import Block, Composite
from sysopt.symbolic import is_symbolic
from sysopt.backends import SymbolicVector
from sysopt.blocks.block_operations import create_functions_from_block
from sysopt.blocks import Gain


class BlockMock(Block):
    args = [(0, 2, 3, 5, 0),
            (1, 1, 7, 5, 1)]
    values = [
        ([[30, ], [2, 3], [-7]],
         [[7*5, ], [1, 7], [1 - 7**2]])
    ]

    def __init__(self, name):
        test_block_metadata = Metadata(
            state=['position'],
            constraints=['constraint'],
            parameters=['rate'],
            inputs=['force'],
            outputs=['position', 'constraint']
        )
        super().__init__(test_block_metadata, name)

    def initial_state(self, parameters: Parameters) -> Numeric:
        return [1]

    def compute_dynamics(self,
                         t: Time,
                         state: States,
                         algebraics: Algebraics,
                         inputs: Inputs,
                         parameters: Parameters):
        x, = state
        z, = algebraics
        u, = inputs
        return [x * z * u]

    def compute_outputs(self,
                        t: Time,
                        state: States,
                        algebraics: Algebraics,
                        inputs: Inputs,
                        parameters: Parameters) -> Numeric:
        x, = state
        z, = algebraics
        return [
            x, z
        ]

    def compute_residuals(self,
                          t: Time,
                          state: States,
                          algebraics: Algebraics,
                          inputs: Inputs,
                          parameters: Parameters) -> Numeric:
        x, = state
        z, = algebraics
        return [x - z**2]



class TestSymbolicFunctionsFromLeafBlock:

    def test_build_functions_from_block(self):
        block_1 = BlockMock("block_1")

        x0, f, g, h = create_functions_from_block(block_1)
        assert g.domain == f.domain == h.domain == (1, 1, 1, 1, 1)
        assert f.codomain == h.codomain == 1, 'Expected 1 output'
        assert g.codomain == 2, 'Expected 2 as per block definition'
        assert x0.domain == 1, 'Expected 1 (p)'
        assert x0.codomain == f.domain.states

    def test_call_functions_numerically(self):

        block_1 = BlockMock("block_1")
        x0, f, g, h = create_functions_from_block(block_1)
        args = (0, 2, 3, 5, 0)
        assert f(*args)[0] == 30, 'Expected block to compute 2 * 3 * 5 == 30'

        assert (g(*args)[0], g(*args)[1]) == (2, 3)
        assert h(*args) == [-7]

    def test_call_functions_symbolically(self):
        block_1 = BlockMock("block_1")
        x0, f, g, h = create_functions_from_block(block_1)
        assert f.domain == g.domain == h.domain
        domain = f.domain
        args = [
            SymbolicVector(name=name, length=domain[i])
            for i, name in enumerate(['t', 'x', 'z', 'u', 'p'])
        ]

        f_result = f(*args)
        assert is_symbolic(f_result)

        g1, g2 = g(*args)
        assert is_symbolic(g1)
        assert is_symbolic(g2)

        h_result = h(*args)
        assert is_symbolic(h_result)

    def test_skip_not_implemented_functions(self):
        # Makes sure we are skipping stuff that isn't defined.
        block = Gain(channels=2)
        x0, f, g, h = create_functions_from_block(block)
        assert not x0
        assert not f
        assert not h

        assert g.domain == (1, 0, 0, 2, 2)
        assert g.codomain == 2


class MockComposite(Composite):
    def __init__(self):
        super().__init__()
        self.block_1 = BlockMock('block_1')
        self.block_2 = BlockMock('block_2')
        self.components = [
            self.block_1, self.block_2
        ]

    @staticmethod
    def args():
        return [1 if count == 0 else (i, j) for count, (i, j) in enumerate(
            zip(*BlockMock.args))]


class TestSymbolicFunctionsFromCompositeBlock:

    def test_composite_functions_are_built(self):
        composite = MockComposite()
        x0, f, g, h = create_functions_from_block(composite)

        assert x0.domain == 2
        assert x0.codomain == 2
        assert f.domain == (1, 2, 2, 2, 2)
        assert f.domain == h.domain

    def test_compose_initial_condition_functions(self):
        composite = MockComposite()
        x0, f, g, h = create_functions_from_block(composite)
        assert x0.domain == 2
        result = x0([1, 1])
        assert len(result) == x0.codomain

        # test composite of composite
        composite2 = Composite()
        composite2.components = [composite, BlockMock('block')]
        x0, f, g, h = create_functions_from_block(composite2)
        assert x0.domain == 3
        assert x0.codomain == 3
        result = x0([1, 1, 1])
        assert len(result) == x0.codomain

    def test_evaluate_composite_initial_conditions_symbolically(self):
        composite = MockComposite()
        x0, f, g, h = create_functions_from_block(composite)
        p = SymbolicVector('p', 2)
        result = x0(p)
        assert len(result) == x0.codomain
        assert is_symbolic(p)
        # test composite of composite
        composite2 = Composite()
        composite2.components = [composite, BlockMock('block')]
        x0, f, _, h = create_functions_from_block(composite2)
        q = SymbolicVector('q', 3)
        result = x0(q)
        assert len(result) == x0.codomain
        assert is_symbolic(q)

    def test_composite_block_functions_numerical_eval(self):
        composite = MockComposite()
        _, f, _, h = create_functions_from_block(composite)

        # computed from block arguments
        args = [1, (2, 1), (3, 7), (5, 5), (0, 1)]
        f_expected = [30, 7*5]
        h_expected = [-7, -48]

        f_result = f(*args)

        assert f_result == f_expected

        h_result = h(*args)
        assert h_result == h_expected

        # test composite of composite
        composite2 = Composite()
        composite2.components = [composite, BlockMock('block')]

        _, f, _, h = create_functions_from_block(composite2)

        # computed from block arguments
        args = [1, (2, 1, 3), (3, 7, 3), (5, 5, 3), (0, 1, 0)]
        f_expected = [30, 7 * 5, 27]

        f_result = f(*args)
        assert f_result == f_expected

    def test_composite_block_functions_symbolic(self):
        composite = MockComposite()
        _, f, g, h = create_functions_from_block(composite)

        # computed from block arguments
        args = [1,
                SymbolicVector('x', 2),
                SymbolicVector('z', 2),
                SymbolicVector('u', 2),
                SymbolicVector('p', 2)
        ]

        f_result = f(*args)
        assert is_symbolic(f_result)
        assert len(f_result) == 2

        # test composite of composite
        composite2 = Composite()
        composite2.components = [composite, BlockMock('block')]
        _, f, _, _ = create_functions_from_block(composite2)

        args = [1,
                SymbolicVector('x', 3),
                SymbolicVector('z', 3),
                SymbolicVector('u', 3),
                SymbolicVector('p', 3)
        ]

        f_result = f(*args)
        assert is_symbolic(f_result)
        assert len(f_result) == 3

    def test_dimensions_of_composite_block_functions_with_wires(self):
        composite = MockComposite()
        composite.wires = [
            (composite.block_1.outputs[0], composite.block_2.inputs),
            (composite.inputs, composite.block_1.inputs),
            (composite.block_2.outputs, composite.outputs)
        ]
        _, f, g, h = create_functions_from_block(composite)
        expected_domain = (1, 2, 3, 1, 2)
        assert f.domain == g.domain == h.domain == expected_domain

        # state hasn't changed
        assert f.codomain == 2

        # each block has two, but we've only set two as outputs
        assert g.codomain == 2

        # each block has 1 constraint, plus the internal connection
        assert h.codomain == 3

    def test_numerical_evaluation_of_block_functions_with_wires(self):
        composite = MockComposite()
        composite.wires = [
            (composite.block_1.outputs[0], composite.block_2.inputs),
            (composite.inputs, composite.block_1.inputs),
            (composite.block_2.outputs, composite.outputs)
        ]
        x0, f, g, h = create_functions_from_block(composite)

        result = x0([1, 1])
        assert len(result) == 2
        assert result == [1, 1]

        args = [1, (2, 1), (3, 7, 3), (5,), (0, 1, 0)]
        assert f.domain == (1, 2, 3, 1, 2)
        f_expected = [30, 21]
        f_result = f(*args)
        assert f_result == f_expected

        g_expected = [1, 7]
        g_result = g(*args)
        assert g_result == g_expected
