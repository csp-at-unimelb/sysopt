import numpy as np
import pytest

from sysopt import Metadata
from sysopt.block import Block, Composite
from sysopt.symbolic import ExpressionGraph, get_time_variable, symbolic_vector, Function, Variable
from sysopt.solver import canonical_transform as xform

from sysopt.blocks.common import Gain, LowPassFilter, Oscillator
from sysopt import exceptions
from dataclasses import asdict

from tests.mocks import LinearScalarEquation
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


class TestExampleBlock:
    def test_init(self):
        block = LinearScalarEquation()
        system = xform.flatten_system(block)
        args = (0, 1, 0, 0, [2, 3])

        result = system.initial_conditions(args[-1])

        assert result == [3]
        output = system.output_map(*args)



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

    @staticmethod
    def get_test_data():
        p = [
            3,  # osc frequency
            0,  # osc phase
            5,  # lpf freaq
            0.5 # gain
        ]
        t = 0
        x = 2
        z = None
        u = [1, 1]

        args = (t, x, z, u, p)
        expected_result = (
            0,          # x0
            (1 - 2)/5,  # f = (u_0 - x) / lpf_freq
            0.5,        # g = u_1 * gain
            [0, 2 - 1]  # h = [cos(1) - u_0, x - u_1]
        )

        return args, expected_result


    def test_correct_tables_are_generated(self):
        block = self.create_composite()
        all_blocks = xform.tree_to_list(block)
        tables, domain = xform.create_tables(all_blocks)
        leaves = [b for b in all_blocks if isinstance(b, Block)]

        for leaf in leaves:
            for var_type, size in asdict(leaf.signature).items():
                table_entries = list(filter(
                    lambda x: x.block == str(leaf),
                    tables[var_type]
                ))
                assert len(table_entries) == size
        assert len(tables['wires']) == 2


    def test_construct_constraints_from_wires(self):
        block = self.create_composite()
        all_blocks = xform.tree_to_list(block)
        tables, domain = xform.create_tables(all_blocks)
        arguments = xform.create_symbols_from_domain(domain)

        test_outputs = symbolic_vector('Outputs', 3)

        vector_constraints = xform.create_constraints_from_wire_list(
            tables['wires'], arguments, test_outputs
        )
        test_args = {
            arguments.u: np.array([2, 3]),
            test_outputs: np.array([2, 3, 4])
        }
        result = vector_constraints.call(test_args)
        assert result.shape == (2,) == vector_constraints.shape
        assert result[0] == 0
        assert result[1] == 0

    def test_flattened_initial_conditions(self):
        block = self.create_composite()
        system = xform.flatten_system(block)
        args, (x0_n, f_n, g_n, h_n) = self.get_test_data()
        x0 = system.initial_conditions(args[-1])
        assert x0[0] == x0_n

    def test_flattened_dynamics_and_output(self):
        block = self.create_composite()
        system = xform.flatten_system(block)
        args, (x0_n, f_n, g_n, h_n) = self.get_test_data()
        f = system.vector_field(*args)
        assert f.shape == system.vector_field.shape
        assert f[0] == f_n

        g = system.output_map(*args)
        assert g.shape == system.output_map.shape

        assert g[0] == g_n

    def test_flattened_constraints(self):
        block = self.create_composite()
        system = xform.flatten_system(block)
        args, (x0_n, f_n, g_n, h_n) = self.get_test_data()

        h = system.constraints(*args)


class TestSYS71Bug:

    @staticmethod
    def build_model():
        from sysopt.blocks import FullStateOutput, ConstantSignal
        model = Composite(name='Test Model')
        # build a LQR model
        #
        plant_metadata = Metadata(
            inputs=['u'],
            states=['x_0', 'x_1']
        )
        A = np.array([[0, 1],
                      [-1, 0]], dtype=float)
        B = np.array([[1], [0]])

        def f(t, x, u, _):
            return A @ x + B @ u

        def x0(_):
            return np.array([0, 1])

        plant = FullStateOutput(
            dxdt=f,
            metadata=plant_metadata,
            x0=x0,
            name='plant'
        )
        control = ConstantSignal(['u'])
        model.components = [plant, control]
        model.declare_outputs(['x_0', 'x_1', 'u'])
        model.wires = [
            (control.outputs, plant.inputs),
            (control.outputs, model.outputs['u']),
            (plant.outputs[0:2], model.outputs[0:2]),
        ]
        return model

    def test_tables_are_correct(self):
        model = self.build_model()
        all_blocks = xform.tree_to_list(model)
        leaves = list(filter(lambda b: not isinstance(b, Composite),
                             all_blocks))
        assert len(leaves) == 2
        tables_raw = [
            xform.create_tables_from_block(b) for b in leaves
        ]
        print(tables_raw)
        # assert False

    def test_scenario(self):
        from sysopt.solver import SolverContext
        # Fix for bug SYS-71
        # Problem is that a fan-out system model is throwing errors
        # ie; that when `get_symbolic_integrator` is called, it
        # is unable to co
        model = self.build_model()
        constants = {p: 1 for p in model.parameters}

        with SolverContext(model, t_final=1, constants=constants) as context:
            y = model.outputs(1)
            constraint = [
                y[0:2].T @ y[0:2] < 1e-9
            ]
            f = context.get_symbolic_integrator()
            # throws here!

