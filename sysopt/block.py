"""Base classes for block-based modelling."""

import weakref
from sysopt.types import (Signature, Metadata, Time, States, Parameters,
                          Inputs, Algebraics, Numeric)
from typing import Iterable, Optional, Union, NewType, Tuple

Pin = NewType('Pin', Union['Port', 'Channel'])
Connection = NewType('Connection', Tuple[Pin, Pin])


class Port:
    """Holds a unique identifier for an input/output port on a block.

    Args:
        parent: The owning object.
        size: The number of channels this port has.

    """

    def __init__(self, parent: 'Block', size: int = 0):
        self._parent = weakref.ref(parent)
        self.size = size
        """(int) Number of 'channel' on this port"""

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
            return item.reference is self
        except AttributeError:
            return item is self

    def __getitem__(self, item):
        assert isinstance(item, int), \
            f'Can\'t get a lazy ref for [{self.parent} {item}]'

        self.size = max(item + 1, self.size)

        return Channel(self, item)

    def __iter__(self):
        return iter(range(self.size))


class Channel:
    """A channel on the associated port."""
    def __init__(self, port: Port, index: int):
        self.port = port
        self.index = index

    @property
    def size(self):
        return 1

    @property
    def parent(self):
        return self.port.parent

    def __iter__(self):
        return iter([self.index])


class Block:
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
                 metadata_or_signature: Union[Signature, Metadata]):

        if isinstance(metadata_or_signature, Metadata):
            self.signature = metadata_or_signature.signature

            self.metadata = metadata_or_signature
        else:
            self.signature = metadata_or_signature
            self.metadata = None

        self.inputs.size = self.signature.inputs
        self.outputs.size = self.signature.outputs

    def __new__(cls, *args, **kwargs):  # noqa
        obj = super().__new__(cls)
        obj.inputs = Port(obj)
        obj.outputs = Port(obj)
        setattr(obj, '__hash__', lambda arg: id(obj))
        return obj

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


class Composite(Block):  # noqa
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
                 wires: Optional[Iterable[Connection]] = None
                 ):
        # pylint: disable=super-init-not-called
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
