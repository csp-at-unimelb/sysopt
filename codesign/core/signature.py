# Intermediate Representation Tables:
#
#  Parameters
#  ----------
#  Constants:
#       uuid, value
#
#  Decision Variables (per optimisation iteration) :
#       uuid, value
#
#   Relations:
#       uuid, expression graph (involving only parameters)
#
#   Slack Constraints:
#       uuid, uuid
#
#
# Signals
# -------
#  Inputs (per optimisation iteration, per integration step):
#       uuid, value
#
#  Outputs (per optimisation iteration, per integration step):
#       uuid, value
#
#  State Variables (per optimisation iteration, per integration step)
#       uuid, value, velocity
#
#  Relations (involving parameters and signals):
#       uuid, expression graph
#
#  Dynamic Slack Constraints:
#       uuid, uuid
#

# Expression Graph Nodes
# ----------------------
#   - elementry n-ary operations
#   - black boxes
#   - curated boxed

# Optimising a System
# -------------------
#   -
#

# loss should be a function of
#
# Constraints should be specified as
# graph(Object) <= 0
# which becomes graph(object) + s = 0
# s >= 0

from dataclasses import dataclass
from typing import Union

from codesign.core.tree_base import  Numeric


@dataclass
class Datum:
    inputs: Union[Numeric, None]
    state: Union[Numeric, None]
    outputs: Union[Numeric, None]
    parameters: Union[Numeric, None]


@dataclass
class Signature:
    inputs: int = 0
    state: int = 0
    outputs: int = 0
    parameters: int = 0

    def __add__(self, other: 'Signature'):
        s = Signature()
        s += self
        s += other
        return s

    def __iadd__(self, other):
        self.inputs += other.inputs
        self.outputs += other.outputs
        self.state += other.state
        self.parameters += other.parameters
        return self

    def __iter__(self):
        return iter((self.inputs, self.state, self.outputs, self.parameters))

