import weakref
from sysopt.types import Signature, Metadata
from typing import Iterable, Optional, Union, NewType, Tuple

Pin = NewType('Pin', Union['LazyReference', 'LazyReferenceChild'])
Connection = NewType('Connection', Tuple[Pin, Pin])


class LazyReference:
    def __init__(self, parent, size=0):
        self._parent = weakref.ref(parent)
        self.size = size

    @property
    def parent(self):
        return self._parent()

    def __len__(self):
        return self.size

    def __hash__(self):
        return id(self)

    def __contains__(self, item):
        try:
            return item.reference is self
        except AttributeError:
            return item is self

    def __getitem__(self, item):
        assert isinstance(item, int), \
            f"Can't get a lazy ref for [{self.parent} {item}]"

        self.size = max(item + 1, self.size)

        return LazyReferenceChild(self, item)

    def get_iterator(self):
        return range(self.size)


class LazyReferenceChild:
    def __init__(self, parent_ref, index):
        self.reference = parent_ref
        self.index = index
        self.size = 1

    @property
    def parent(self):
        return self.reference.parent

    def get_iterator(self):
        return iter([self.index])


class Block:
    def __init__(self,
                 signature: Union[Signature, Metadata]):

        if isinstance(signature, Metadata):
            self.signature = signature.signature
            self.metadata = signature
        else:
            self.signature = signature
            self.metadata = None

        self.inputs.size = self.signature.inputs
        self.outputs.size = self.signature.outputs
        self.state.size = self.signature.state
        self.parameters.size = self.signature.parameters
        self.constraints.size = self.signature.constraints

    def __new__(cls, *args, **kwargs):
        obj = super(Block, cls).__new__(cls)
        obj.inputs = LazyReference(obj)
        obj.outputs = LazyReference(obj)
        obj.state = LazyReference(obj)
        obj.parameters = LazyReference(obj)
        obj.constraints = LazyReference(obj)

        return obj

    def uuid(self):
        return id(self)

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        return None

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return None

    def compute_residuals(self, t, state, algebraics, inputs, parameters):
        return None

    def initial_state(self, parameters):
        return None

    def call(self, t, state, algebraic, inputs, parameters):
        return (
            self.compute_dynamics(t, state, algebraic, inputs, parameters),
            self.compute_outputs(t, state, algebraic, inputs, parameters),
            self.compute_residuals(t, state, algebraic, inputs, parameters)
        )


class ConnectionList(list):
    def __init__(self, parent):
        super().__init__()
        self._parent = weakref.ref(parent)

    def __iadd__(self, other):
        for pair in other:
            self.add(pair)

    @property
    def parent(self):
        return self._parent()

    def add(self, pair):
        src, dest = pair
        if not src.size and dest.size:
            src.size = dest.size
        elif not dest.size and src.size:
            dest.size = src.size
        elif not src.size and not dest.size:
            raise ConnectionError(f"Cannot connect {src} to {dest}, "
                                  f"both have unknown dimensions")
        elif src.size != dest.size:
            raise ConnectionError(f"Cannot connect {src} to {dest}, "
                                  f"incomaptible dimensions")
        self.append((src, dest))


class Composite(Block):
    def __init__(self,
                 components: Optional[Iterable[Block]] = None,
                 wires: Optional[Iterable[Connection]] = None
                 ):
        self._wires = ConnectionList(self)

        self.components = components or []      # type: Iterable[Block]
        self.wires = wires or []                # type: Iterable[Connection]

    @property
    def wires(self):
        return self._wires

    @wires.setter
    def wires(self, value):
        if isinstance(value, list):
            self._wires.clear()
            for pair in value:
                self._wires.add(pair)
        elif value is self._wires:
            return
