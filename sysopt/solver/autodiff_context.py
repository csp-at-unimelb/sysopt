from sysopt.symbolic import SymbolicVector, concatenate, list_symbols, lambdify
from sysopt.block import Block
from sysopt.solver.flattened_system import FlattenedSystem
from sysopt.optimisation import DecisionVariable


class ADContext:
    """Autodiff context"""

    def __init__(self, t_final=1):
        self.t_final = t_final
        self._t = SymbolicVector('t', 1)
        self._signals = {}
        self._outputs = {}
        self._model_variables = {}
        self._free_variables = {}

        if isinstance(t_final, DecisionVariable):
            self._free_variables['T'] = t_final
        self.path_variables = []
        self.expressions = []

    def get_or_create_signals(self, block: Block):
        """Creates or retrieves the symbolic variables for the given block."""
        assert not hasattr(block, 'components'), \
            f"cannot create signal for composite {block}"
        try:
            return self._signals[block]
        except KeyError:
            pass
        n = block.signature.state
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
            f = concatenate(*f) if f is not None else None
            g = concatenate(*g) if g is not None else None
            h = concatenate(*h) if h is not None else None
            x0 = concatenate(*x0) if x0 is not None else None
        except RuntimeError as ex:
            raise ValueError(
                f'Could not stack functions form block {block}:'
                'Are you sure they\'re returning a list or tuple?'
            ) from ex

        return FlattenedSystem(*variables, f, g, h, X0=x0)

    def _recursively_flatten(self, block: Block):

        # pylint: disable=invalid-name
        try:
            flattened_systems = []
            uuids = {}
            for i, component in enumerate(block.components):
                flattened_systems.append(self._recursively_flatten(component))
                uuids[component] = i

        except AttributeError as ex:
            return self._flatten_leaf(block)
        offset_map = {}

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
            X=concatenate(*x_flat),
            U=concatenate(*U_flat),
            Z=concatenate(*z_flat, *z_new),
            P=concatenate(*p_flat),
            f=concatenate(*f_flat),
            g=concatenate(*g_flat),
            h=concatenate(*h_flat, *h_new),
            X0=concatenate(*x0_flat)
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

    def _get_or_create_block_decision_variables(self,
                                                name, block, indices
                                                ):

        if block not in self._model_variables:
            self._model_variables[block] = {}

        new_vars = {
            i: SymbolicVector(f'{name}_{i}')
            for i in indices
            if i not in self._model_variables[block]
        }
        self._model_variables[block].update(new_vars)

        return concatenate(
            *(self._model_variables[block][i] for i in indices)
        )

    def get_or_create_decision_variable(self,
                                        name: str,
                                        block=None,
                                        parameter=None):
        if not block:
            try:
                var = self._free_variables[name]
            except KeyError:
                var = SymbolicVector(name, 1)
                self._free_variables[name] = var
        elif parameter is None:
            var = self._get_or_create_block_decision_variables(
                name, block, list(range(block.signature.parameters))
            )

        else:
            try:
                idx = block.metadata.parameters.index(parameter)
            except ValueError as ex:
                if 0 <= parameter < block.signature.parameters:
                    idx = parameter
                else:
                    msg = f"Invalid parameter identified: {parameter}"
                    raise ValueError(msg) from ex
            var = self._get_or_create_block_decision_variables(
                name, block, [idx]
            )

        return var

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
        x, z, u, p = self.get_or_create_signals(block)
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
