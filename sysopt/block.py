"""Base classes for block-based modelling."""
from typing import Iterable, Optional, Union, NewType, Tuple, List
import weakref
from dataclasses import  asdict
from sysopt.types import (Signature, Metadata, Time, States, Parameters,
                          Inputs, Algebraics, Numeric)

Pin = NewType('Pin', Union['Port', 'Channel'])
Connection = NewType('Connection', Tuple[Pin, Pin])


class Port:
    """Holds a unique identifier for an input/output port on a block.

    Args:
        parent: The owning object.
        size: The number of channels this port has.

    """

    def __init__(self, port_type, parent: 'Block', size: int = 0):
        self._parent = weakref.ref(parent)
        self.port_type = port_type
        self._channels = []
        self.size = size
        """(int) Number of 'channel' on this port"""

    def __str__(self):
        return f'{str(self.parent)}->{self.port_type}'

    @property
    def size(self):
        return len(self._channels)

    @size.setter
    def size(self, value):
        difference = value - self.size
        if difference >= 0:
            offset = self.size
            self._channels += [
                Channel(self, [i + offset]) for i in range(difference)
            ]

    @property
    def parent(self):
        """The block that this port is from."""
        return self._parent()

    def __len__(self):
        return self.size

    def __hash__(self):
        return id(self)

    def __contains__(self, item):
        try:
            return item.port is self
        except AttributeError:
            return item is self

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.size = max(item.stop, self.size)
            indicies = list(
                range(item.start, item.stop,
                      item.step if item.step else 1)
            )
            return Channel(self, indicies)

        elif isinstance(item, int):
            self.size = max(item + 1, self.size)
            return Channel(self, [item])
        elif isinstance(item, str):
            idx = self.parent.find_by_name(self.port_type, item)
            if idx >= 0:
                return Channel(self, [idx])
        raise ValueError(f'Can\'t get a lazy ref for [{self.parent} {item}]')

    def __iter__(self):
        return iter(self._channels)

    def __call__(self, t):
        ctx = t.context
        return ctx.signal(self, list(range(self.size)), t)


class Channel:
    """A channel on the associated port."""
    def __init__(self, port: Port, indices: List[int]):
        self.port = port
        self.indices = indices

    def __call__(self, t):
        ctx = t.context
        return ctx.signal(self.port, self.indices, t)

    @property
    def size(self):
        return len(self.indices)

    @property
    def parent(self):
        return self.port.parent

    def __iter__(self):
        return iter(self.indices)

class ComponentBase:
    __instance_count = 0

    def __init__(self, name=None):
        self.__instance_count += 1
        self.name = name or f'{type(self).__name__}_{self.__instance_count}'

    @property
    def parent(self):
        if self._parent:
            return self._parent()
        return None

    @parent.setter
    def parent(self, value):
        if value is None:
            self._parent = None
        else:
            assert isinstance(value,  Composite)
            self._parent = weakref.ref(value)

    def trunk(self):
        node = self
        tree = []
        while node is not None:
            tree.append(node)
            node = node.parent
        return reversed(tree)

    def __str__(self):
        return '/'.join([node.name for node in self.trunk()])

    def compute_dynamics(self,
                         t: Time,
                         state: States,
                         algebraics: Algebraics,
                         inputs: Inputs,
                         parameters: Parameters):
        raise NotImplementedError

    def compute_outputs(self,
                        t: Time,
                        state: States,
                        algebraics: Algebraics,
                        inputs: Inputs,
                        parameters: Parameters) -> Numeric:
        raise NotImplementedError

    def compute_residuals(self,
                          t: Time,
                          state: States,
                          algebraics: Algebraics,
                          inputs: Inputs,
                          parameters: Parameters) -> Numeric:
        raise NotImplementedError

    def initial_state(self, parameters: Parameters) -> Numeric:
        raise NotImplementedError

    def __new__(cls, *args, name=None, **kwargs):  # noqa
        obj = super().__new__(cls)
        obj.inputs = Port('inputs', obj)
        obj.outputs = Port('outputs', obj)
        setattr(obj, '__hash__', lambda arg: id(obj))
        setattr(obj, '_parent', None)
        return obj


class Block(ComponentBase):
    r"""Base class for component models.

    Blocks represent the fundamental components in a model, and
    describe parameter dynamics and/or input-output maps.

    A block is made up of input, output, parameter, state and
    algebraically constrained spaces.
    The dimension of these spaces are defined either implicitly by
    metadata, or explicitly via an instance `sysopt.Signature`.
    Formally, we define these spaces as :math:`U,Y,P,X,Z` respectively.
    Then, a system can be characterised by the functions $f,g,h$ such
    that

    math::
        \dot{x} = f(t, x, z, u, p)
              y = g(t, x, z, y, p)
              0 = h(t, x, z, y, p)

    and with initial conditions :math:`x0(p) = x(0; p)` when it is relevant.

    Args:
        metadata_or_signature: The metadata or signature describing the
            dimensions of the funadmental spaces of this system.

    Attributes:
        signature: An instance of `sysopt.Signature` describing the dimensions
            of input, state, algebraic, output and parameter spaces.
        metadata: An optional instance of `sysopt.Metadata`
            describing the metadata (eg. names) of each term in the input,
            output, state, algebraic and parameter spaces.
        inputs: An instance of `Port` used to define connections.
        outputs: An instance of `Port` used to define connections.

    """

    def __init__(self,
                 metadata_or_signature: Union[Signature, Metadata],
                 name=None
                 ):

        if isinstance(metadata_or_signature, Signature):
            metadata_or_signature = Metadata.from_signature(
                metadata_or_signature
            )

        self.metadata = metadata_or_signature

        self.inputs.size = self.signature.inputs
        self.outputs.size = self.signature.outputs
        super().__init__(name)

    @property
    def parameters(self):
        name = str(self)
        return [ f"{name}/{p}" for p in self.metadata.parameters]

    def find_by_name(self, var_type, name):
        try:
            values = asdict(self.metadata)[var_type]
        except KeyError as ex:
            msg = f"{var_type} is not a valid metadata field"
            raise ValueError(msg) from ex
        return values.index(name)

    @property
    def signature(self):
        return self.metadata.signature



class ConnectionList(list):
    """Container for connections between ports.

    Args:
        parent: Composite object that contains this connection list.

    """

    def __init__(self, parent: 'Composite'):
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
            raise ConnectionError(f'Cannot connect {src} to {dest}, '
                                  f'both have unknown dimensions')
        elif src.size != dest.size:
            raise ConnectionError(f'Cannot connect {src} to {dest}, '
                                  f'incompatible dimensions')
        self.append((src, dest))


class ListIndexableByName(list):
    def __getitem__(self, item):
        if isinstance(item, str):
            idx = self.find_by_name(item)
            if idx < 0:
                raise KeyError(f"{item} is not in list")

            return self[idx]
        else:
            return super().__getitem__(item)

    def find_by_name(self, name):
        index = -1
        for i, item in enumerate(self):
            if str(item) == name:
                return i
        return index


class Composite(ComponentBase):  # noqa
    """Block that consists of a sub-blocks and connections.

    Instances of `sysopt.Block` can be added to a composite block.
    Wires between ports can then be specified to enforce flow between
    different sub-blocks.

    Wires can also define 'forwarding' relationships, for example
    between inputs from the composite block, into inputs on sub-blocks.

    This allows models to be constructed hierarchically, and provides a
    means for encapsulation.

    Args:
        components: A list of components
        wires: A list of connections between component ports.

    """

    def __init__(self,
                 components: Optional[Iterable[Block]] = None,
                 wires: Optional[Iterable[Connection]] = None,
                 name=None
                 ):
        super().__init__(name)
        # pylint: disable=super-init-not-called
        self._wires = ConnectionList(self)
        self._components = []
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

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, values):
        for item in values:
            item.parent = self
            self._components.append(item)

    @property
    def parameters(self):

        return [p for sub_block in self.components
                for p in sub_block.parameters]