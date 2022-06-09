import numpy as np
import pytest

from sysopt import Metadata
from sysopt.block import Block, Composite
from sysopt.symbolic import ExpressionGraph, get_time_variable, symbolic_vector, Function
from sysopt.solver import canonical_transform as xform

from sysopt.blocks.common import Gain, LowPassFilter, Oscillator
from sysopt import exceptions
from dataclasses import asdict

md = Metadata(
    inputs=['u_0', 'u_1'],
    outputs=['y'],
    states=['x_0', 'x_1'],
    constraints=[],
    parameters=['p']
)

eye = np.eye(2)

class MockBlockCorrect(Block):

    def __init__(self):
        super().__init__(md, name='test_block')

    def initial_state(self, parameters):
        return [parameters[0], 0]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        return - eye @ states + inputs

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states[0] + states[1]

    def get_symbolic_args(self):
        return xform.Arguments(
            get_time_variable(),
            symbolic_vector('x', self.signature.states),
            symbolic_vector('z', self.signature.constraints),
            symbolic_vector('u', self.signature.inputs),
            symbolic_vector('p', self.signature.parameters)
        )

    def get_numerical_args(self):
        return 0, [1,2], 3, [5, 7], 11


class MockBlockIncorrect(Block):

    def __init__(self):
        super().__init__(md, name='test_block')

    def initial_state(self, parameters):
        return [parameters[0], 0, 0]

    def compute_dynamics(self, t, states, algebraics, inputs, parameters):
        raise NotImplementedError

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return [states[0] + states[1], 1]

    def get_symbolic_args(self):
        return xform.Arguments(
            get_time_variable(),
            symbolic_vector('x', self.signature.states),
            symbolic_vector('z', self.signature.constraints),
            symbolic_vector('u', self.signature.inputs),
            symbolic_vector('p', self.signature.parameters)
        )



class TestLeafBlockXform:
    def test_correct_tables_are_generated(self):
        block = MockBlockCorrect()
        tables = xform.create_tables_from_blocks(block)
        for var_type, size in asdict(block.signature).items():
            assert len(tables[var_type]) == size

    def test_correct_creation_and_evaluation_of_initial_coniditions(self):
        block = MockBlockCorrect()
        args = block.get_symbolic_args()
        x0 = xform.symbolically_evaluate_initial_conditions(block, args)

        value = x0(11)
        assert value == [11, 0], 'Numerical Evaluation failed.'

    def test_correct_creation_an_evaluation_of_dynamics(self):
        block = MockBlockCorrect()
        args = block.get_symbolic_args()
        f = xform.symbolically_evaluate(
            block, block.compute_dynamics, block.signature.states, args
        )

        t,x,z,u,p = block.get_numerical_args()

        result = f(x, u)
        expected_result = [5 - 1, 7 - 2]
        assert result == expected_result

    def test_incorrect_size_ic_should_throw(self):
        block = MockBlockIncorrect()
        args = block.get_symbolic_args()
        with pytest.raises(exceptions.FunctionError):
            x0 = xform.symbolically_evaluate_initial_conditions(
                block, args
            )


    def test_incorrect_size_outputs_should_throw(self):
        block = MockBlockIncorrect()
        args = block.get_symbolic_args()
        with pytest.raises(exceptions.FunctionError):
            f = xform.symbolically_evaluate(
                block, block.compute_outputs, block.signature.outputs, args
            )

    def test_not_implemented_function_throws(self):
        block = MockBlockIncorrect()
        args = block.get_symbolic_args()
        with pytest.raises(exceptions.FunctionError):
            f = xform.symbolically_evaluate(
                block, block.compute_dynamics, block.signature.states, args
            )


class TestComposite:

    @staticmethod
    def create_composite():
        osc = Oscillator()
        gain = Gain(channels=1)
        lpf = LowPassFilter()
        composite = Composite()
        composite.components = [osc, lpf, gain]
        composite.wires = [
            (osc.outputs, lpf.inputs),
            (lpf.outputs, gain.inputs),
            (gain.outputs, composite.outputs)
        ]
        return composite


    def test_correct_tables_are_generated(self):
        block = self.create_composite()
        all_blocks = xform.tree_to_list(block)
        tables, domain  = xform.create_tables(all_blocks)
        leaves = [b for b in all_blocks if isinstance(b, Block)]

        for leaf in leaves:
            for var_type, size in asdict(leaf.signature).items():
                table_entries = list(filter(
                    lambda x: x.block == str(leaf),
                    tables[var_type]
                ))
                assert len(table_entries) == size
        assert len(tables['wires']) == 2


    # def test_initial_conditions_are_correctly_mapped(self):
    #     block = self.create_composite()
    #     tables = xform.create_tables(block)
    #     lpf, = filter(lambda x: isinstance(x, LowPassFilter), block.components)
    #     arguments = xform.create_symbols_from_tables(tables)
    #
    #     x0, f, g, h = xform.symbolically_evaluate_block(tables, lpf, arguments)
    #

    def test_flattening(self):
        block = self.create_composite()
        system = xform.flatten_system(block)
        
        assert False

