"""Base classes for block-based modelling."""
from typing import Iterable, Optional, Union, NewType, Tuple, List
import weakref
from dataclasses import asdict
from sysopt.types import (Signature, Metadata, Time, States, Parameters,
                          Inputs, Algebraics, Numeric)

from sysopt.symbolic.symbols import SignalReference, restriction_map
from sysopt.exceptions import DanglingInputError, InvalidWire, UnconnectedInputError


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

    def __repr__(self):
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
            indices = list(
                range(item.start, item.stop,
                      item.step if item.step else 1)
            )
            return Channel(self, indices)

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
        y_t = SignalReference(self)

        if t is y_t.t:
            return y_t
        else:
            return y_t(t)

    @property
    def indices(self) -> List[int]:
        return list(range(self.size))


class Channel:
    """A channel on the associated port."""
    def __init__(self, port: Port, indices: List[int]):
        self.port = port
        self.indices = indices  # type: List[int]

    @property
    def port_type(self):
        return self.port.port_type

    def __call__(self, t):
        y = self.port(t)

        pi = restriction_map(self.indices, y.shape[0])
        return pi(y)

    @property
    def size(self):
        return len(self.indices)

    @property
    def parent(self):
        return self.port.parent

    def __iter__(self):
        return iter(self.indices)


class ComponentBase:
    """Interface definition and recursive search methods for components."""
    _instance_count = 0  # pylint: disable=invalid-name

    def __init__(self, *args, **kwargs):
        pass

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

    def __repr__(self):
        return '/'.join([node.name for node in self.trunk()])

    def compute_dynamics(self,
                         t: Time,
                         states: States,
                         algebraics: Algebraics,
                         inputs: Inputs,
                         parameters: Parameters):
        raise NotImplementedError

    def compute_outputs(self,
                        t: Time,
                        states: States,
                        algebraics: Algebraics,
                        inputs: Inputs,
                        parameters: Parameters) -> Numeric:
        raise NotImplementedError

    def compute_residuals(self,
                          t: Time,
                          states: States,
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
        ComponentBase._instance_count += 1
        setattr(
            obj, 'name',
            name or f'{cls.__name__}_{ComponentBase._instance_count}'
        )

        return obj


class Block(ComponentBase):
    r"""Base class for component models.

    Blocks represent the fundamental components in a model, and
    describe parameter dynamics and/or input-output maps.

    A block is made up of input, output, parameter, states and
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
            of input, states, algebraic, output and parameter spaces.
        metadata: An optional instance of `sysopt.Metadata`
            describing the metadata (eg. names) of each term in the input,
            output, states, algebraic and parameter spaces.
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
        if not self.metadata.parameters:
            return []

        name = str(self)
        return [f'{name}/{p}' for p in self.metadata.parameters]

    def find_by_name(self, var_type, name):
        try:
            values = asdict(self.metadata)[var_type]
        except KeyError as ex:
            msg = f'{var_type} is not a valid metadata field'
            raise ValueError(msg) from ex
        return values.index(name)

    @property
    def signature(self):
        return self.metadata.signature

    def find_by_type_and_name(self, var_type, var_name: str):
        block_name = str(self)
        if var_name.startswith(f'{block_name}/'):
            name = var_name[len(block_name) + 1:]

            index = self.find_by_name(var_type, name)
            if index >= 0:
                return self, index

        return None


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
        if src is dest:
            raise ConnectionError(f'Cannot connect {src} to {dest}, '
                                  'both arguments are the same')
        if not src.size and dest.size:
            src.size = dest.size
        elif not dest.size and src.size:
            dest.size = src.size
        elif not src.size and not dest.size:
            raise ConnectionError(
              f'Cannot connect {src} to {dest}, '
              f'both have unknown dimensions. '
              f'Error occurs in Composite {self._parent()} '
              f'when connecting blocks {src.parent} to {dest.parent}.')
        elif src.size != dest.size:
            raise ConnectionError(
              f'Cannot connect {src} to {dest}, '
              f'incompatible dimensions. '
              f'Error occurs in Composite {self._parent()} '
              f'when connecting blocks {src.parent} to {dest.parent}.')
        self.append((src, dest))


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
        valid_components = {self} | set(self._components)
        if isinstance(value, list):
            self._wires.clear()
            for pair in value:
                src, dest = pair
                if src.parent not in valid_components:
                    raise InvalidWire('Failed to add wires:'
                                      f'source component {src.parent} '
                                      f'not found for wire {pair}'
                                      f'Error arises in composite {self}')
                if dest.parent not in valid_components:
                    raise InvalidWire('Failed to add wires:'
                                      f'Sink component {dest.parent} '
                                      f'not found for wire {pair}'
                                      f'Error arises in composite {self}')
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
        return [
            p for sub_block in self.components for p in sub_block.parameters
        ]

    def find_by_type_and_name(self, var_type, var_name):
        for component in self.components:
            result = component.find_by_type_and_name(var_type, var_name)
            if result:
                return result

        return None
        

def validate_inputs(composite: Composite):
    external_inputs = {channel for channel in composite.inputs}
    external_outputs = {channel for channel in composite.outputs}
    internal_inputs = {
        channel for component in composite.components
        for channel in component.inputs
    }

    for src, dest in composite.wires:

        if isinstance(src, Port):
            def src_filter(channel):
                return channel.port == src
        else:
            def src_filter(channel):
                return channel == src

        if isinstance(dest, Port):
            def dest_filter(channel):
                return channel.port == dest
        else:
            def dest_filter(channel):
                return channel == dest

        external_inputs = set(filter(src_filter, external_inputs))
        internal_inputs = set(filter(dest_filter, internal_inputs))
        external_outputs = set(filter(dest_filter, external_outputs))

    if external_inputs:
        raise DanglingInputError(
            str(composite), {str(channel) for channel in external_inputs}
        )
    if internal_inputs:
        raise UnconnectedInputError(
            str(composite), {str(channel) for channel in internal_inputs}
        )
