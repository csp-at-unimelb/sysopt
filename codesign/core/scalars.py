import abc
from numbers import Number
from typing import Set, Tuple, Optional
from codesign.core.tree_base import Algebraic
from codesign.core.name_registry import register_or_create_name, register_default_name


class Atomic(metaclass=abc.ABCMeta):
    def __init__(self, name=None):
        self.name = register_or_create_name(self, name)

    def atoms(self) -> Set['Atomic']:
        return {self}


@register_default_name('x')
class Variable(Algebraic, Atomic):
    def __hash__(self):
        return id(self)


@register_default_name('p')
class Parameter(Algebraic, Atomic):
    def __init__(self,
                 value: Number = 0,
                 bounds: Optional[Tuple[float, float]]=None):
        super().__init__()
        self.value = value
        self.bounds = bounds

    def __hash__(self):
        return id(self)
