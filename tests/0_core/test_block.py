import numpy as np

from sysopt import Signature, Block
from sysopt.blocks import InputOutput

# Variable
# - name
# - bounds[optional]

# SIVariable(Variable)
# - units


# Signature
# - states
# - constraints
# - inputs
# - outputs
# - parameters


# Metadata
# - states: List[Union[str, Variable]]
# - constraints
# - inputs

# ----------------Test Fixtures---------------------
class BlockMock(Block):
    def __init__(self):
        super().__init__(
            signature=Signature(inputs=2, outputs=2)
        )

    def compute_outputs(self, t, x, z, inputs, parameters):
        return t * inputs


def mock_block_factory():
    sig = Signature(inputs=2, outputs=2)

    def g(t, u, _):
        return t * u

    return InputOutput(sig, g)


# ------------------- Tests
def test_block_oop_api():
    block = BlockMock()

    assert block.signature == Signature(
        inputs=2, outputs=2, parameters=0, state=0, constraints=0
    )

    assert len(block.inputs) == 2
    assert len(block.outputs) == 2
    assert len(block.parameters) == 0
    assert len(block.state) == 0
    assert len(block.constraints) == 0


def test_block_factory():
    block = mock_block_factory()

    arg = np.array([1, 3], dtype=float)
    T = 2
    truth = np.array([2, 6], dtype=float)

    result = block.compute_outputs(T, None, None, arg, None)

    assert (result - truth == 0).all()
