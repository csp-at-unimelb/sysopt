from typing import Callable, Optional
from sysopt.types import Numeric


class TimePoint:
    pass


class IntegrationWindow:
    def __init__(self, t_final):
        self.__t_final = t_final

    @property
    def stop(self):
        return self.__t_final

    @property
    def start(self) -> float:
        return 0

    @property
    def t(self):
        return TimePoint()


def integral(function: Callable[[float], Numeric]) \
        -> Callable[[float], Numeric]:
    raise NotImplementedError


class DecisionVariable:
    pass


class Minimise:
    pass


def solve(problem):
    pass
