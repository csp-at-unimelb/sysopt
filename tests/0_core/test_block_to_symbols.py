from sysopt.types import *
from sysopt.block import Block, Composite
from sysopt.symbolic import create_functions_from_block, SymbolicVector, is_symbolic
from sysopt.blocks import Gain


class BlockMock(Block):
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
        assert f(*args) == [30, ], 'Expected block to compute 2 * 3 * 5 == 30'
        assert g(*args) == [2, 3]
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

        f_result, = f(*args)
        assert is_symbolic(f_result)

        g1, g2 = g(*args)
        assert is_symbolic(g1)
        assert is_symbolic(g2)

        h_result, = h(*args)
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


class TestSymbolicFunctionsFromCompositeBlock:

    def test_composite_functions_are_built(self):
        composite = MockComposite()
        x0, f, g, h = create_functions_from_block(composite)

        assert x0.domain == 2
        assert x0.codomain == 2
        assert f.domain == (1, 2, 2, 2, 2)
        assert g.domain == f.domain == h.domain

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
        assert False

    def test_composite_block_functions_dimensions(self):
        assert False

    def test_composite_block_functions_numerical_eval(self):
        assert False

    def test_composite_block_functions_numerical_symbolic(self):
        assert False

    # We're done one we've got these.
    # Then, we need to make sure we can differentiate them.
    
