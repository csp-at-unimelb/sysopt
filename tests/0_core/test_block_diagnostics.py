import pytest
from sysopt.block import check_wiring_or_raise, Composite
from sysopt.exceptions import UnconnectedInputError
from sysopt.blocks.common import Gain, Oscillator


def test_invalid_wiring():
    osc = Oscillator()
    gain = Gain(channels=1)

    model = Composite()
    model.components = [osc, gain]

    with pytest.raises(UnconnectedInputError):
        check_wiring_or_raise(model)


def test_valid_wiring():
    osc = Oscillator()
    gain = Gain(channels=1)

    model = Composite()
    model.components = [osc, gain]
    model.wires = [
        (osc.outputs, gain.inputs),
        (gain.outputs, model.outputs)
    ]
    check_wiring_or_raise(model)





