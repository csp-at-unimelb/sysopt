"""Sympy Backend Implementation.

This backend is designed to provide symbolic equations for use in other codes.
It does not contain ODE solvers and other features of the Casade backend.
"""

import sympy as sp
import numpy as np
import pickle 

from sysopt.backends.sympy.math import fmin, fmax
from sysopt.backends.sympy.symbols import *
from sysopt.symbolic.casts import cast_like

class InterpolatedPath:
    """Function that linearly interpolates between the data-points."""
    def __init__(self, t, x):
        self.t = t
        self.x = x

    @property
    def t_max(self):
        return self.t[-1]

    def __call__(self, t):
        """Get the value at `t`, assumed between 0 and T_max."""
        for i in range(self.t.shape[1] - 1):
                dist = self.t[i + 1] - self.t[i]
                w0 = (self.t[i + 1] - t)/dist
                w1 = (t - self.t[i])/dist
                return w0 * self.x[:, i] + w1*self.x[:, i+1]

        raise ValueError(f'No data point for {t}')


def list_symbols(expr) -> set:
    return expr.free_symbols

def lambdify(expressions, arguments, name='f'):
    # CasADI api - throws general exception
    # pylint: disable=broad-except
    try:
        outputs = [concatenate(expr) for expr in expressions]
    except Exception:
        outputs = [expressions]
    return sp.lambdify(arguments,outputs)

class KO_inputs:
    def __init__(self,backend,model):
        
        print("Flattening system...")
        flat_model = backend.get_flattened_system(model)
        print("Substituting unique symbol names...")
        flat_model = backend.unique_sym_names(flat_model)
        print("Solving explicit symbol links...")
        flat_model = backend.sub_explicit_eqs(flat_model)
        #  print("Simplifying expressions...")
        #  flat_model = backend.factor_eqs(flat_model)
        backend.create_signal_dicts()
        self.X = flat_model.X
        self.U = flat_model.U
        self.Z = flat_model.Z
        self.P = flat_model.P
        self.f = flat_model.f
        self.g = flat_model.g
        self.h = flat_model.h

        self.sym_dict = backend.sym_dict
        self.sym_dict_inv = backend.sym_dict_inv

        self.state_dict = backend.state_dict
        self.input_dict = backend.input_dict
        self.param_dict = backend.param_dict

        self.state_dict_inv = backend.state_dict_inv
        self.input_dict_inv = backend.input_dict_inv
        self.param_dict_inv = backend.param_dict_inv

    def save(self,name):
        file = open(name, 'wb')
        pickle.dump(self,file)
        file.close()
