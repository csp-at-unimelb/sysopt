"""Sympy Backend Implementation.

This backend is designed to provide symbolic equations for use in other codes.
It does not contain ODE solvers and other features of the Casade backend.
"""

import sympy as sp
import numpy as np

from sysopt.backend import ADContext, FlattenedSystem
from sysopt.block import Block

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
            if self.t[i] <= t <= self.t[i + 1]:
                dist = self.t[i + 1] - self.t[i]
                w0 = (self.t[i + 1] - t)/dist
                w1 = (t - self.t[i])/dist
                return w0 * self.x[:, i] + w1*self.x[:, i+1]

        raise ValueError(f'No data point for {t}')

class SympyVector(sp.Matrix):
    """Wrapper around sympy.symbols for vectors."""
    def __init__(self):
        super().__init__()

    def __new__(cls, name, length):
        assert isinstance(length, int)
        names = [name+"_"+str(i) for i in range(length)]
        symbols_ = sp.symbols(names)
        obj=sp.Matrix(symbols_)
        # TODO: Confirm whether vector needs to be tranposed
        # obj = sp.Transpose(sp.Matrix(symbols_))
        # obj.__class__ = SympyVector
        return obj

    def __iter__(self):
        return iter(
            [self[i] for i in range(self.shape[0])]
        )


class SympyBackend(ADContext):
    """Autodiff context based on the casadi library."""

    name = 'SymPy'

    def __init__(self):
        self._t = sp.Symbol('t')
        self._variables = {}

        self._nx = 0
        self._nu = 0
        self._nz = 0
        self.table = []

    def wrap_function(self, function, *args):
        raise NotImplementedError

    def get_or_create_variables(self,
                                block: Block):
        """Creates or retrieves the symbolic variables for the given block."""
        assert not hasattr(block, 'components')
        try:
            return self._variables[block]
        except KeyError:
            pass
        n = block.signature.state
        m = block.signature.constraints
        k = block.signature.inputs
        ell = block.signature.parameters

        x = SympyVector('x', n) if n > 0 else None
        z = SympyVector('z', m) if m > 0 else None
        u = SympyVector('u', k) if k > 0 else None
        p = SympyVector('p', ell) if ell > 0 else None
        variables = (x, z, u, p)
        self._variables[block] = variables
        return variables

    def cast(self, arg):
        # TODO: I don't think casting is needed for Sympy
        # if arg is None:
            # return None
        # if isinstance(arg, (float, int)):
            # return casadi.SX(arg)
        if not isinstance(arg, sp.Matrix):
            return sp.Matrix([arg])
        
        # if isinstance(arg, (list, tuple, np.ndarray)):
            # return sp.Matrix(arg)

        raise NotImplementedError(f'Don\'t know how to cast {arg.__class__}')

    @property
    def t(self):
        return self._t

    def concatenate(self, *vectors):
        # Should be working for Sympy
        """Concatenate arguments into a sympy symbolic vector."""
        try:
            v0, *v_n = vectors
        except ValueError:
            return None
        while v0 is None:
            try:
                v0, *v_n = v_n
            except ValueError:
                return None
        if not isinstance(v0, sp.Matrix):
            result = self.cast(v0)
        else:
            result = v0
        for v_i in v_n:
            if v_i is not None:
                if not isinstance(v_i, sp.Matrix):
                    v_i = self.cast(v_i)
                result = result.col_join(v_i)
        return result


    def get_flattened_system(self, block: Block):
        flat_system = self._recursively_flatten(block)

        return flat_system

    def _flatten_leaf(self, block: Block):

        # pylint: disable=invalid-name

        variables = self.get_or_create_variables(block)
        t = self.t
        try:
            f = block.compute_dynamics(t, *variables)
            x0 = block.initial_state(variables[-1])
        except NotImplementedError:
            f, x0 = None, None
        try:
            g = block.compute_outputs(t, *variables)
        except NotImplementedError:
            g = None
        try:
            h = block.compute_residuals(t, *variables)
        except NotImplementedError:
            h = None
        try:
            f = self.concatenate(*f) if f is not None else None
            g = self.concatenate(*g) if g is not None else None
            h = self.concatenate(*h) if h is not None else None
            x0 = self.concatenate(*x0) if x0 is not None else None
        except RuntimeError as ex:
            raise ValueError(
                f'Could not stack functions form block {block}:'
                'Are you sure they\'re returning a list or tuple?'
            ) from ex

        return FlattenedSystem(
            *variables, f, g, h, X0=x0
        )

    def nlp(self, system: FlattenedSystem):
        raise NotImplementedError

    # def integrator(self, system: FlattenedSystem):
        # if system.X is None:
            # return self.nlp(system)
        # if system.U is not None:
            # raise ValueError('System has unassigned inputs')
        # return CasadiOdeSolver(self.t, system)

    def _recursively_flatten(self, block: Block):

        # pylint: disable=invalid-name
        try:
            flattened_systems = []
            uuids = {}
            for i, component in enumerate(block.components):
                flattened_systems.append(self._recursively_flatten(component))
                uuids[component] = i

        except AttributeError:
            return self._flatten_leaf(block)

        x_flat, z_flat, p_flat, f_flat, h_flat, x0_flat = zip(*[
            (subsys.X, subsys.Z, subsys.P, subsys.f, subsys.h, subsys.X0)
            for subsys in flattened_systems
        ])
        U_dict = {}
        g_dict = {}
        h_new = []
        z_new = []
        for src, dest in block.wires:
            if src in block.inputs:
                U_dict.update(
                    dict(zip(src, self._get_input_symbols_for(dest)))
                )
            elif dest in block.outputs:
                idx = uuids[src.parent]
                g_idx = flattened_systems[idx].g
                g_dict.update({
                    i: g_idx[j]
                    for i, j in zip(src, dest)
                })
            else:
                idx = uuids[src.parent]
                g_idx = flattened_systems[idx].g
                symbols = self._get_input_symbols_for(dest)
                h_new += [
                    u_i - g_idx[j]
                    for u_i, j in zip(symbols, src)
                ]
                z_new += list(self._get_input_symbols_for(dest))

        U_flat = [
            u_i for _, u_i in sorted(U_dict.items(), key=lambda item: item[0])
        ]

        g_flat = [
            g_i for _, g_i in sorted(g_dict.items(), key=lambda item: item[0])
        ]
        return FlattenedSystem(
            X=self.concatenate(*x_flat),
            U=self.concatenate(*U_flat),
            Z=self.concatenate(*z_flat, *z_new),
            P=self.concatenate(*p_flat),
            f=self.concatenate(*f_flat),
            g=self.concatenate(*g_flat),
            h=self.concatenate(*h_flat, *h_new),
            X0=self.concatenate(*x0_flat)
        )

    def _get_input_symbols_for(self, lazy_reference):
        block = lazy_reference.parent
        _, _, u, _ = self.get_or_create_variables(block)

        if lazy_reference in block.inputs:
            return iter(u[i] for i in lazy_reference)

        raise ValueError(
            f'Can\'t get input symbols for {lazy_reference.parent}'
        )
