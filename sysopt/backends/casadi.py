import casadi
import numpy as np

from sysopt.backend import CodesignSolverContext, FlattenedSystem
from sysopt.types import Signature


class CasadiOdeSolver:
    def __init__(self, t, system: FlattenedSystem):
        assert system is not None
        assert system.X is not None
        assert system.f is not None
        self._T = casadi.SX.sym('T', 1, 1)
        dae_spec = {
            'x': casadi.vertcat(t, system.X),
            'p': casadi.vertcat(self._T, system.P),
            'ode': self._T * casadi.vertcat(casadi.SX.ones(1, 1), system.f),
        }
        self.x0 = casadi.Function(
            'x0',
            [system.P],
            [casadi.vertcat(casadi.SX.zeros(1, 1), system.X0)]
        )
        self.g = casadi.Function(
            'g',
            [t,  system.X, system.Z, system.P],
            [system.g]
        )
        self.n_alg = 0
        if system.Z is not None:
            dae_spec.update({
                'z': system.Z,
                'alg': system.h
            })
            self.n_alg, _ = system.Z.shape

        self.dae_spec = dae_spec

    def __call__(self, t, p):
        """Integrate from 0 to t"""
        n = 50
        solver_options = {
            'grid': [i / n for i in range(n + 1)],
            'output_t0': True
        }
        integrator = casadi.integrator('F', 'idas', self.dae_spec, solver_options)

        p_prime = casadi.vertcat(t, p)
        x0 = self.x0(p)
        z0 = [0] * self.n_alg

        soln = integrator(x0=x0, p=p_prime, z0=z0)

        tf = soln['xf'][0, :]
        x = soln['xf'][1:, :]
        z = soln['zf']

        y = self.g(tf, x, z, p)

        return InterpolatedPath(tf, y)


class InterpolatedPath:
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def __call__(self, t):
        for i in range(self.t.shape[1] - 1):
            if self.t[i] <= t <= self.t[i + 1]:
                dist = self.t[i + 1] - self.t[i]
                w0 = (self.t[i + 1] - t)/dist
                w1 = (t - self.t[i])/dist
                return w0 * self.x[:, i] + w1*self.x[:, i+1]

        raise ValueError(f"No data point for {t}")


class CasadiVector(casadi.SX):
    def __init__(self, *args, **kwarg):
        pass

    def __new__(cls, name, length):
        assert isinstance(length, int)
        obj = CasadiVector.sym(name, length)
        obj.__class__ = CasadiVector
        return obj

    def __iter__(self):
        return iter(
            [self[i] for i in range(self.shape[0])]
        )


class CasadiBackend(CodesignSolverContext):
    name = 'CasADI'

    def __init__(self):
        self._t = casadi.SX.sym('t', 1)
        self._variables = {}

        self._nx = 0
        self._nu = 0
        self._nz = 0
        self.table = []

    def wrap_function(self, function, *args):
        raise NotImplementedError

    def get_or_create_variables(self, block):
        assert not hasattr(block, 'components')
        try:
            return self._variables[block.uuid()]
        except KeyError:
            pass
        n = block.signature.state
        m = block.signature.constraints
        k = block.signature.inputs
        ell = block.signature.parameters

        x = CasadiVector('x', n) if n > 0 else None
        z = CasadiVector('z', m) if m > 0 else None
        u = CasadiVector('u', k) if k > 0 else None
        p = CasadiVector('p', ell) if ell > 0 else None
        variables = (x, z, u, p)
        self._variables[block.uuid()] = variables
        return variables

    def cast(self, arg):
        if arg is None:
            return None
        if isinstance(arg, (float, int)):
            return casadi.SX(arg)
        if isinstance(arg, (casadi.SX, casadi.MX, casadi.DM)):
            return arg
        elif isinstance(arg, (list, tuple, np.ndarray)):
            return casadi.SX(arg)
        else:
            raise NotImplementedError(f"Don't know how to cast {arg.__class__}")

    @property
    def t(self):
        return self._t

    def concatenate(self, *vectors):
        try:
            v0, *v_n = vectors
        except ValueError:
            return None
        while v0 is None:
            try:
                v0, *v_n = v_n
            except ValueError:
                return None
        if not isinstance(v0, casadi.SX):
            result = self.cast(v0)
        else:
            result = v0
        for v_i in v_n:
            if v_i is not None:
                result = casadi.vertcat(result, v_i)

        return result

    def get_flattened_system(self, block):
        flat_system = self._recursively_flatten(block)

        return flat_system

    def _flatten_leaf(self, block):

        variables = self.get_or_create_variables(block)
        t = self.t
        f = block.compute_dynamics(t, *variables)
        g = block.compute_outputs(t, *variables)
        h = block.compute_residuals(t, *variables)
        x0 = block.initial_state(variables[-1])
        print(x0)
        try:
            f = self.concatenate(*f) if f is not None else None
            g = self.concatenate(*g) if g is not None else None
            h = self.concatenate(*h) if h is not None else None
            x0 = self.concatenate(*x0) if x0 is not None else None
        except RuntimeError:
            raise ValueError(f"Could not stack functions form block {block}: "
                             "Are you sure they're returning a list or tuple?")

        return FlattenedSystem(
            *variables, f, g, h, X0=x0
        )

    def nlp(self, system: FlattenedSystem):
        raise NotImplementedError

    def integrator(self, system: FlattenedSystem):
        if system.X is None:
            return self.nlp(system)
        if system.U is not None:
            raise ValueError(f"System has unassigned inputs")
        return CasadiOdeSolver(self.t, system)

    def _recursively_flatten(self, block):
        try:
            flattened_systems = []
            uuids = {}
            for i, component in enumerate(block.components):
                flattened_systems.append(self._recursively_flatten(component))
                uuids[component.uuid()] = i

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
                U_dict.update({
                    i: u
                    for i, u in zip(src.get_iterator(),
                                    self._get_input_symbols_for(dest))
                })
            elif dest in block.outputs:
                idx = uuids[src.parent.uuid()]
                g_idx = flattened_systems[idx].g
                g_dict.update({
                    i: g_idx[j] for i, j in zip(src.get_iterator(), dest.get_iterator())
                })
            else:
                idx = uuids[src.parent.uuid()]
                g_idx = flattened_systems[idx].g
                symbols = self._get_input_symbols_for(dest)
                h_new += [
                    u_i - g_idx[j]
                    for u_i, j in zip(symbols, src.get_iterator())
                ]
                z_new += list(self._get_input_symbols_for(dest))
        U_flat = [u_i for _, u_i in sorted(U_dict.items(), key=lambda item: item[0])]
        g_flat = [g_i for _, g_i in sorted(g_dict.items(), key=lambda item: item[0])]
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
            return iter(u[i] for i in lazy_reference.get_iterator())

        raise ValueError(f"Can't get input symbols for {lazy_reference.parent}")
