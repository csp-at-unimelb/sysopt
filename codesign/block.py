from codesign.core import Signature, Vector, Atomic, Parameter, Variable, Number

from typing import Iterable, List, Set, Optional, Union, NewType, Tuple

from dataclasses import dataclass


@dataclass
class Metadata:
    inputs: List[str]
    outputs: List[str]
    state: List[str]
    parameters: List[str]

    @property
    def signature(self):
        return Signature(
            inputs=len(self.inputs),
            outputs=len(self.outputs),
            state=len(self.state),
            parameters=len(self.parameters)
        )


class Block:
    def __init__(self,
                 signature: Signature,
                 parameters: Iterable[Union[Parameter, Number]] = None):
        self.signature = signature
        self.inputs = Vector.create_filled_with(Variable, signature.inputs)
        self.state = Vector.create_filled_with(Variable, signature.state)
        self.outputs = Vector.create_filled_with(Variable, signature.outputs)

        params_to_add = signature.parameters
        param_list = []
        try:
            param_list += list(parameters)
        except TypeError:
            pass

        params_to_add -= len(param_list)
        new_params = [Parameter() for _ in range(params_to_add)]
        self.parameters = Vector(param_list + new_params)

    def atoms(self) -> Set[Atomic]:
        result = set()
        for vector in (self.inputs, self.outputs, self.state, self.parameters):
            try:
                atoms = vector.atoms()
            except AttributeError:
                continue
            result |= atoms
        return result

    def expressions(self) -> Tuple[Vector, Vector, Vector]:
        """
        Returns vector expressions (f, g, h) such that
            \dot{x} = f(t, x,u, p)
            y = g(t, x, u)
            0 = h(t, x, u)

        """
        raise NotImplementedError


class Problem:
    def __init__(self, model, constraints=None, loss=None):
        self.model = model

    def solve(self):
        raise NotImplemented


Connection = NewType(
    'Connection', Union[Tuple[Variable, Variable],
                        Tuple[Iterable[Variable], Iterable[Variable]]])


class System:
    def __init__(self,
                 components: Optional[Iterable[Block]] = None,
                 wires: Optional[Iterable[Connection]] = None):
        self.components = components or []  # type: Iterable[Block]
        self.wires = wires or []            # type: Iterable[Connection]

    def atoms(self):
        result = set()
        for component in self.components:
            result |= component.atoms()
        return result

    @property
    def signature(self):
        signature = Signature()
        for component in self.components:
            signature += component.signature
        return signature

    def dynamic_expressions(self) -> Vector:
        raise  NotImplementedError

    def constraint_expressions(self) -> Vector:
        raise NotImplementedError

    def output_expressions(self) -> Vector:
        raise NotImplementedError

