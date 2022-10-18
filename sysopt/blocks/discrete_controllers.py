"""Discrete Time Controllers.

Includes:
 - PID Controller

"""

from typing import List, Optional
from sysopt import Metadata
from sysopt.modelling.block import DiscreteBlock

from sysopt.symbolic import concatenate


class PIDController(DiscreteBlock):
    """Discrete time PID controller."""

    def __init__(self,
                 error_signals: Optional[List[str]] = None,
                 clock_hz=100,
                 name=None):

        if error_signals is None:
            error_signals = ['0']

        outputs = [f'y^{sig}' for sig in error_signals]

        delays = [f'{line_id}{sig}'
                  for line_id in ('y', 'e_-1', 'e_-2')
                  for sig in error_signals]

        gains = [f'{sig}:{p}'
                 for sig in error_signals
                 for p in ('K_d', 'K_p', 'K_i')]

        metadata = Metadata(
            inputs=error_signals,
            states=delays,
            outputs=outputs,
            parameters=gains
        )

        super().__init__(
            metadata_or_signature=metadata,
            clock_hz=clock_hz,
            name=name
        )

    def initial_state(self, parameters):
        return [0] * self.signature.states

    def compute_state_transition(
        self, t, states, algebraics, inputs, parameters
    ):

        dt = 1 / self.frequency

        # PID Controller
        # states = u[k-1], e[k - 1], e[k - 2]
        # inputs = e[k]
        # parameters = [K^0_d, K^0_p, K^0_i, K^1_p, ... ]

        # state transition computes
        # u[k] = u[k-1]
        #        + (K_p + K_i * dt + K_d / dt) e[k]
        #        + (-K_p - 2K_d/dt) e[k - 1]
        #        + (K_d/dt) * e[k - 2]

        diff_gain = parameters[0::3]
        prop_gain = parameters[1::3] * self.frequency
        intg_gain = parameters[2::3] * dt
        u_last = states[0::3]
        e_last = states[1::3]
        e_lastlast = states[2::3]
        u_next = (
             u_last + (prop_gain + intg_gain + diff_gain) * inputs
            + (prop_gain + 2 * diff_gain) * e_last
            + prop_gain * e_lastlast
        )

        return concatenate(u_next, inputs, e_last)

    def compute_outputs(self, t, states, algebraics, inputs, parameters):
        return states[0::3]
