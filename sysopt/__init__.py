"""sysopt - Component-base system modelling and optimisation.

Build component based models, symbolically evaluate them, and translate the
result into desired backend format.

Packages:
    modelling - Classes and methods for constructing component based models.
    problems  - API for solving model simulation and optimisation problems.
    symbolic  - Classes and methods for building symbolic expressions.
    backends  - Interface shims for different algorithmic/numerical routines.

"""

from sysopt.env import version
from sysopt.modelling import *
from sysopt.symbolic import *
from sysopt.problems import *
