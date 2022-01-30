

class Block:
    def __init__(self, state_size, input_size):
        pass

    def call(self, dx, x, u, t, p):
        """

        Return:
            output, residual

        """
        raise NotImplemented


class Function(Block):
    def __init__(self, input_size):
        super(Function).__init__(state_size=0, input_size=input_size)

    def call(self, u, t, p):
        return 0




def test_pid():

    model = new_model()

    chain = sequential(
        stack,
        filter,
        parallel(controller1, controller2),
        collapse,
        dynamics
    )
    feedback_chain = sequential(
        dynamics,
        stack
    )
