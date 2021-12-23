from typing import List, Tuple, NewType
Expression = NewType('Expression', None)


class DesignParameter:
    def __init__(self, value, minimum=None, maximum=None):
        self.min = minimum
        self.max = maximum
        self.value = value


class Model:
    def variables(self) -> Tuple[List[str]]:
        raise NotImplemented

    def initial_state(self):
        if self.variables()[0]:
            raise NotImplemented
        else:
            return None

    def get_residual(self, dt, dx, t, x, u, y, p) -> List[Expression]:
        raise NotImplemented

    def get_system_constraints(self, t, x, u, y, p):
        return None

    def get_parameter_constraints(self, p):
        raise None

    @property
    def signature(self):
        return [len(name_list) for name_list in self.variables()]
