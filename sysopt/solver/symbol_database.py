"""Symbol Database for simulation and optimisation."""
# pylint: disable=invalid-name
from typing import Iterable, Callable, Optional, NamedTuple, Union
from dataclasses import dataclass


from sysopt.backends import (
    SymbolicVector, concatenate_symbols, list_symbols,
    lambdify
)
from sysopt.symbolic import Variable
from sysopt.block import Block


BoundParameter = NamedTuple(
    'BoundParameter',
    [('block', Block), ('parameter', Union[int, str])]
)



@dataclass
class FlattenedSystem:
    """Intermediate representation of a systems model."""
    X: Optional[Iterable] = None            # Dynamics
    Z: Optional[Iterable] = None            # Coupling Variables
    U: Optional[Iterable] = None            # Inputs
    P: Optional[Iterable] = None            # Parameters
    f: Optional[Callable] = None            # Explicit Dynamics
    g: Optional[Callable] = None            # Outputs
    h: Optional[Callable] = None            # Algebraic Constraints.
    j: Optional[Callable] = None            # Quadratures
    X0: Optional[Iterable] = None           # Initial values

    def __iadd__(self, other):
        assert isinstance(other, FlattenedSystem)
        self.X = concatenate_symbols(self.X, other.X)
        self.Z = concatenate_symbols(self.Z, other.Z)
        self.U = concatenate_symbols(self.U, other.U)
        self.P = concatenate_symbols(self.P, other.P)
        self.f = concatenate_symbols(self.f, other.f)
        self.g = concatenate_symbols(self.g, other.g)
        self.h = concatenate_symbols(self.h, other.h)
        self.j = concatenate_symbols(self.j, other.j)
        self.X0 = concatenate_symbols(self.X0, other.X0)

        return self

    def __add__(self, other):
        result = FlattenedSystem()
        result += self
        result += other
        return result

class SymbolDatabase:
    """Autodiff context"""

    def __init__(self, t_final=1):
        self.t_final = t_final
        self._t = SymbolicVector('t', 1)
        self._signals = {}
        self._outputs = {}
        self._model_variables = {}
        self._free_variables = {}

        if isinstance(t_final, Variable):
            self._free_variables['T'] = t_final
        self.path_variables = []
        self.expressions = []

    def get_or_create_signals(self, block: Block):
        """Creates or retrieves the symbolic variables for the given block."""
        assert not hasattr(block, 'components'), \
            f'cannot create signal for composite {block}'
        try:
            return self._signals[block]
        except KeyError:
            pass
        n = block.signature.states
        m = block.signature.constraints
        k = block.signature.inputs
        ell = block.signature.parameters

        x = SymbolicVector('x', n) if n > 0 else None
        z = SymbolicVector('z', m) if m > 0 else None
        u = SymbolicVector('u', k) if k > 0 else None
        p = SymbolicVector('p', ell) if ell > 0 else None
        variables = (x, z, u, p)
        self._signals[block] = variables
        return variables

    @property
    def t(self):
        return self._t

    def get_flattened_system(self, block: Block):
        flat_system = self._recursively_flatten(block)

        return flat_system



    def _flatten_leaf(self, block: Block):

        # pylint: disable=invalid-name

        variables = self.get_or_create_signals(block)
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
            f = concatenate_symbols(*f) if f is not None else None
            g = concatenate_symbols(*g) if g is not None else None
            h = concatenate_symbols(*h) if h is not None else None
            x0 = concatenate_symbols(*x0) if x0 is not None else None
        except RuntimeError as ex:
            raise ValueError(
                f'Could not stack functions form block {block}:'
                'Are you sure they\'re returning a list or tuple?'
            ) from ex

        return FlattenedSystem(*variables, f, g, h, X0=x0)

    def factor_eqs(self,flat_system):
        import sympy as sp

        for i in range(len(flat_system.f)):
            flat_system.f[i] = sp.simplify(flat_system.f[i])
        return flat_system

    def sub_explicit_eqs(self,flat_system):

        import sympy as sp

        Z = flat_system.Z
        h = flat_system.h

        repl = [(Z[i],Z[i]-h[i]) for i in range(len(Z))]

        for i in range(len(flat_system.f)):
            j = 0
            while not flat_system.f[i].free_symbols.isdisjoint(Z):
                flat_system.f[i] = flat_system.f[i].subs(repl)
                j += 1
                if j> 10:
                    print("Error: Hit recusive limit hit for variable substitution")
                    print("Cause is likely because of incorrect wiring")

        #  h_ = sp.solve(h,Z,quick=True,minimal=True,simplify=False)
        #  repl = [(Z[i],h_[Z[i]]) for i in range(len(Z))]
        #  for i in range(len(flat_system.f)):
            #  flat_system.f[i] = flat_system.f[i].subs(repl)

        self.sym_vars = set()
        for f in flat_system.f:
            self.sym_vars |= f.free_symbols
        self.sym_vars |= set(flat_system.X)

        return flat_system

    def unique_sym_names(self,flat_system):
        n_x = len(flat_system.X) if flat_system.X != None else 0
        n_z = len(flat_system.Z) if flat_system.Z != None else 0
        n_u = len(flat_system.U) if flat_system.U != None else 0
        n_p = len(flat_system.P) if flat_system.P != None else 0

        X_o = [flat_system.X[i] for i in range(n_x)]
        P_o = [flat_system.P[i] for i in range(n_p)]
        U_o = [flat_system.U[i] for i in range(n_u)]
        Z_o = [flat_system.Z[i] for i in range(n_z)]

        X_n = SymbolicVector('x',n_x,Dummy=False)
        P_n = SymbolicVector('p',n_p,Dummy=False)
        U_n = SymbolicVector('u',n_u,Dummy=False)
        Z_n = SymbolicVector('z',n_z,Dummy=False)

        X_n = [X_n[i] for i in range(n_x)]
        P_n = [P_n[i] for i in range(n_p)]
        U_n = [U_n[i] for i in range(n_u)]
        Z_n = [Z_n[i] for i in range(n_z)]

        var_o = X_o + P_o + U_o + Z_o
        var_n = X_n + P_n + U_n + Z_n

        repl = [(var_o[i],var_n[i]) for i in range(len(var_o))]
        #  for i in range(n_x):
        flat_system.X = flat_system.X.subs(repl) if flat_system.X != None else []
        flat_system.P = flat_system.P.subs(repl) if flat_system.P != None else []
        flat_system.U = flat_system.U.subs(repl) if flat_system.U != None else []
        flat_system.Z = flat_system.Z.subs(repl) if flat_system.Z != None else []
        flat_system.f = flat_system.f.subs(repl) if flat_system.f != None else []
        flat_system.g = flat_system.g.subs(repl) if flat_system.g != None else []
        flat_system.h = flat_system.h.subs(repl) if flat_system.h != None else []

        for key,value in self._signals.items():
            _signals = list(value)
            for i in range(len(_signals)):
                if _signals[i] != None:
                    _signals[i] = _signals[i].subs(repl)

            self._signals[key] = tuple(_signals)
        return flat_system

    def list_blocks(self, block:Block):

        if not hasattr(block, 'components'):
            return block

        blocks = []
        for component in block.components:
            _blocks = self.list_blocks(component)
            if isinstance(_blocks,list):
                blocks += _blocks
            else:
                blocks.append(self.list_blocks(component))

        self.blocks = blocks
        return blocks

    def create_signal_dicts(self):

        state_dict = {}
        input_dict = {}
        parameter_dict = {}
        for block, signals in self._signals.items():
            if block.metadata.states:
                for i,name in enumerate(block.metadata.states):
                    if signals[0][i] in self.sym_vars:
                        state_dict[signals[0][i]] = name
            if block.metadata.inputs:
                for i,name in enumerate(block.metadata.inputs):
                    if signals[2][i] in self.sym_vars:
                        input_dict[signals[2][i]] = name
            if block.metadata.parameters:
                for i,name in enumerate(block.metadata.parameters):
                    if signals[3][i] in self.sym_vars:
                        parameter_dict[signals[3][i]] = name

        self.sym_dict = {**state_dict, **input_dict, **parameter_dict}

        self.state_dict = state_dict
        self.input_dict = input_dict
        self.param_dict = parameter_dict

        self.sym_dict_inv = {v: k for k, v in self.sym_dict.items()}

        self.state_dict_inv = {v: k for k, v in self.state_dict.items()}
        self.input_dict_inv = {v: k for k, v in self.input_dict.items()}
        self.param_dict_inv = {v: k for k, v in self.param_dict.items()}

        if len(self.sym_dict) != len(self.sym_dict_inv):
            print("WARNING: multiple identical parameters/inputs found. This will cause errors")
            seen = set()
            dupes = [x for x in list(self.sym_dict.values()) if x in seen or seen.add(x)]
            print("The following values are dupliates:")
            for d in dupes:
                print(d)


        return 

    def _recursively_flatten(self, block: Block):

        # pylint: disable=invalid-name
        if not hasattr(block, 'components'):
            return self._flatten_leaf(block)

        flattened_systems = []
        uuids = {}
        for i, component in enumerate(block.components):
            flattened_systems.append(self._recursively_flatten(component))
            uuids[component] = i

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
                g_dict.update({j: g_idx[i] for i, j in zip(src, dest)})
            else:

                idx = uuids[src.parent]
                g_idx = flattened_systems[idx].g

                symbols = self._get_input_symbols_for(dest)

                h_new += [u_i - g_idx[j] for u_i, j in zip(symbols, src)]
                z_new += list(self._get_input_symbols_for(dest))

        U_flat = [
            u_i for _, u_i in sorted(U_dict.items(), key=lambda item: item[0])
        ]
        g_flat = [
            g_i for _, g_i in sorted(g_dict.items(), key=lambda item: item[0])
        ]

        return FlattenedSystem(
            X=concatenate_symbols(*x_flat),
            U=concatenate_symbols(*U_flat),
            Z=concatenate_symbols(*z_flat, *z_new),
            P=concatenate_symbols(*p_flat),
            f=concatenate_symbols(*f_flat),
            g=concatenate_symbols(*g_flat),
            h=concatenate_symbols(*h_flat, *h_new),
            X0=concatenate_symbols(*x0_flat)
        )

    def _get_input_symbols_for(self, lazy_reference):
        block = lazy_reference.parent
        _, _, u, _ = self.get_or_create_signals(block)

        if lazy_reference in block.inputs:
            return iter(u[i] for i in lazy_reference)

        raise ValueError(
            f'Can\'t get input symbols for {lazy_reference.parent}'
        )

    def get_or_create_outputs(self, block):
        try:
            y = self._outputs[block]
        except KeyError:
            y = SymbolicVector('y', block.outputs.size)
            self._outputs[block] = y
        return y

    def list_variables(self, expression_or_equation):
        block_parameters = {
            param
            for param_list in self._model_variables.values()
            for param in param_list.values()
        }
        return {
            var for var in list_symbols(expression_or_equation)
            if var in set(self._free_variables.values()) | block_parameters
        }

    def get_or_create_port_variables(self, port):
        if port is port.parent.inputs:
            _, _, u, _ = self.get_or_create_signals(port.parent)
            return u
        if port is port.parent.outputs:
            return self.get_or_create_outputs(port.parent)

    def get_parameter_offset(self, flattened_system, block):
        *_, p = self.get_or_create_signals(block)
        return flattened_system.P.index(p)

    def get_path_variable(self, expression, args, name='v'):
        assert isinstance(name, str)
        s = SymbolicVector(name, expression.shape[0])
        self.path_variables.append(
            (s, lambdify(expression, [args], name))
        )
        return s

    def get_point_variable(self, expression, t, args, name='s'):
        assert isinstance(name, str)
        s = SymbolicVector(name, expression.shape[0])
        self.expressions.append(
            (s, t, lambdify(expression, [args], name))
        )
        return s
