"""Sysopt Warning Messages"""


class UnconnectedInput(Warning):
    def __init__(self, wire):

        msg = f'Skipping wire {str(wire[0])} -> {str(wire[1])} as ' \
              f'input {wire[1]} has not been forwarded to any components.'
        super().__init__(msg)
