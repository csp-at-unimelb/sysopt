from typing import Iterable
from .interfaces import Function, Block


class Passthrough(Function):
    pass


class Sequential(Block):
    def __init__(self, blocks:Iterable[Block], name=None):
        super(Sequential).__init__(name)


class Stack(Function):
    _n = 0
    def __init__(self, blocks:Iterable[Block], name=None):
        super(Stack).__init__(name)
        self.blocks = blocks



