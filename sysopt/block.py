import weakref
from sysopt.types import Signature, Metadata
from typing import Iterable, List, Optional, Union, NewType, Tuple

Pin = NewType('Pin', Tuple['LazyReference', int])
Connection = NewType(
    'Connection', Union[Tuple['LazyReference', 'LazyReference'], Tuple[Pin, Pin]]
)


class LazyReference:
    def __init__(self, parent, size=0):
        self._parent = weakref.ref(parent)
        self.size = size

    @property
    def parent(self):
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
        assert isinstance(item, int), f"Can't get a lazy ref for {self.parent}" \
                                      f"{item}"

        self.size = max(item + 1, self.size)

        return LazyReferenceChild(self, item)

    def get_iterator(self):
        return range(self.size)


class LazyReferenceChild:
    def __init__(self, parent_ref, index):
        self.reference = parent_ref
        self.index = index
        self.size = 1

    @property
    def parent(self):
        return self.reference.parent

    def get_iterator(self):
        return iter([self.index])


class Block:
    def __init__(self,
                 signature: Union[Signature, Metadata]):

        if isinstance(signature, Metadata):
            self.signature = signature.signature
            self.metadata = signature
        else:
            self.signature = signature
            self.metadata = None

        self.inputs.size = self.signature.inputs
        self.outputs.size = self.signature.outputs
        self.state.size = self.signature.state
        self.parameters.size = self.signature.parameters
        self.constraints.size = self.signature.constraints

    def __new__(cls, *args, **kwargs):
        obj = super(Block, cls).__new__(cls)
        obj.inputs = LazyReference(obj)
        obj.outputs = LazyReference(obj)
        obj.state = LazyReference(obj)
        obj.parameters = LazyReference(obj)
        obj.constraints = LazyReference(obj)

        return obj

    def uuid(self):
        return id(self)

    def compute_dynamics(self, t, state, algebraics, inputs, parameters):
        return None

    def compute_outputs(self, t, state, algebraics, inputs, parameters):
        return None

    def compute_residuals(self, t, state, algebraics, inputs, parameters):
        return None

    def initial_state(self, parameters):
        return None

    def call(self, t, state, algebraic, inputs, parameters):
        return (
            self.compute_dynamics(t, state, algebraic, inputs, parameters),
            self.compute_outputs(t, state, algebraic, inputs, parameters),
            self.compute_residuals(t, state, algebraic, inputs, parameters)
        )


class Problem:
    def __init__(self, model, constraints=None, loss=None):
        self.model = model

    def solve(self):
        raise NotImplemented


class VariableMap:
    def __init__(self, parent: 'Composite', attribute: str):
        self._parent = weakref.ref(parent)
        self.attribute = attribute
        self._map = {}
        self._length = 0
        self.rebuild()

    @property
    def parent(self):
        return self._parent()

    def __len__(self):
        return self._length

    def rebuild(self):
        parent = self.parent
        if not parent:
            return

        self._map.clear()
        length = 0
        for component in parent.components:
            n = getattr(component.signature, self.attribute)
            if n == 0:
                continue
            bounds = (length, length + n)
            self._map[component.uuid()] = bounds
            length += n
        self._length = length

    def invert(self, system_variables):
        parent = self.parent
        inverse_map = {}
        for component in parent.components:
            try:
                i, j = self._map[component.uuid()]
            except KeyError:
                continue
            inverse_map[component] = system_variables[i: j]
        return inverse_map

    def apply(self, *args: Tuple[Block, List[float]]):
        system_variables = [0.0] * self._length
        for component, variables in args:
            i, j = self._map[component.uuid()]
            assert len(variables) == j - i, \
                f"Incorrect number of {self.attribute} for {component}:" \
                f"Expected: {j - i}, got {len(variables)}."
            system_variables[i: j] = variables
        return system_variables

#
# Wiring map takes care of
# [U_outer; Z_outer; Y_outer] = J [U_inner; Y_inner; Z_inner]
# [Z_outer, Y_outer] = J_1 [Y_inner]
# [U_inner] = J_2 [Z_outer, U_outer]
# [Z_inner] = J_3 [Z_outer]
#
# J_i are all orthonormal by construction
#             J_i^{-1} = J_i^T


class WiringMap:
    def __init__(self, parent: 'Composite'):
        self._parent = weakref.ref(parent)
        self._inputs = 0
        self._outputs = 0
        self._constraints = 0
        self._ext_inputs = {}
        self._ext_output = {}
        self._int_inputs = {}
        self._int_outputs = {}
        self._algrabiac_map = {}
        self.rebuild()

    @property
    def parent(self):
        return self._parent()

    def size(self):
        return self._inputs, self._outputs, self._constraints

    @staticmethod
    def validate_wire(source, destination):
        if isinstance(source, LazyReference) \
                and isinstance(destination, LazyReference)\
                and len(source) == len(destination):
            return zip(range(len(destination)), range(len(source)))
        try:
            _, src_port = source
            dest, dest_port = destination
            assert isinstance(src_port, int) and isinstance(dest_port, int)
        except (ValueError, AssertionError):
            raise ValueError(
                f"Wire from {source} to {destination} has incompatible shape"
            )

        return dest, [(dest_port, src_port)]

    def rebuild(self):
        parent = self.parent
        if not parent:
            return

        # if an input == an output
        # the input space and output space both go down by 1
        # and the constraint space goes up by 1
        # u_i == y_j  -> u_i == z_k, y_j == z_k
        # and the 'output' becomes an algebraic constraint.

        _out_out_map = {
            # component: (component output, system output)
        }
        _out_constraint_map = {
            # component: (c_output, system algebraic)
        }
        _in_in_map = {
            # component: (component inputs, system_input)
        }
        _in_constraint_map = {
            # component: (component input, system algebraic)
        }
        _constraint_map ={

        }

        n_u = 0
        n_y = 0
        n_z = 0
        for c in parent.components:
            n_constraints = c.signature.constraints
            if n_constraints > 0:
                _constraint_map[c.uuid()] = (n_z, n_z + n_constraints)
                n_z += n_constraints

        for src, dest in parent.wires:
            try:
                comp_0, index_0 = src
                indicies_0 = [index_0]
            except ValueError:
                comp_0 = src
                indicies_0 = list(range(len(comp_0)))
            try:
                comp_1, index_1 = dest
                indicies_1 = [index_1]
            except ValueError:
                comp_1 = dest
                indicies_1 = list(range(len(comp_1)))

            assert len(indicies_0) == len(indicies_1), \
                f"Wire from {comp_0.parent} to {comp_1.parent} have incompatible sizes"

            # case 1: input->input
            if src in parent.inputs:
                for pair in zip(indicies_0, indicies_1):
                    _in_in_map[comp_1.parent.uuid()] = pair
                    n_u += 1
                continue
            elif dest in parent.outputs:
                for pair in zip(indicies_1, indicies_0):
                    _out_out_map[comp_0.parent.uuid()] = pair
                    n_y += 1
                continue
            elif src in comp_0.parent.outputs and dest in comp_1.parent.inputs:
                for out_port, in_port in zip(indicies_0, indicies_1):
                    z_idx = n_z
                    n_z += 1
                    _out_constraint_map[comp_0.parent.uuid()] = (out_port, z_idx)
                    _in_constraint_map[comp_1.parent.uuid()] = (in_port, z_idx)
            else:
                raise NotImplementedError(
                    f"Don't know how to handle wires from {src} to {dest}"
                )
        self._inputs = n_u
        self._outputs = n_y
        self._constraints = n_z
        self._ext_inputs = _in_in_map
        self._int_inputs = _in_constraint_map
        self._ext_output = _out_out_map
        self._int_outputs = _out_constraint_map
        self._algrabiac_map = _constraint_map

    def invert_inputs(self, inputs, algebraic):
        input_map = {}
        algebraic_map = {}
        for component in self.parent.components:
            size = component.signature.inputs
            if size == 0:
                continue
            component_inputs = [0] * size
            try:
                for i, j in self._ext_inputs[component.uuid()]:
                    component_inputs[j] = inputs[i]
            except (KeyError, TypeError):
                pass
            try:
                for i, j in self._int_inputs[component.uuid()]:
                    component_inputs[j] = algebraic[i]
            except (KeyError, TypeError):
                pass

            try:
                start, end = self._algrabiac_map[component.uuid()]
                algebraic_map[component] = algebraic[start:end]
            except KeyError:
                algebraic_map[component] = None
            input_map[component] = component_inputs

        return input_map, algebraic_map

    def apply_outputs(self, component, component_outputs,
                      system_outputs, system_residuals
                      ):
        raise NotImplementedError

    def apply_residuals(self, component, component_residuals,
                      system_outputs, system_residuals
                      ):
        raise NotImplementedError


class ConnectionList(list):
    def __init__(self, parent):
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
            raise ConnectionError(f"Cannot connect {src} to {dest}, "
                                  f"both have unknown dimensions")
        elif src.size != dest.size:
            raise ConnectionError(f"Cannot connect {src} to {dest}, "
                                  f"incomaptible dimensions")
        self.append((src, dest))


class Composite(Block):
    def __init__(self,
                 components: Optional[Iterable[Block]] = None,
                 wires: Optional[Iterable[Connection]] = None
                 ):
        self._wires = ConnectionList(self)

        self.components = components or []      # type: Iterable[Block]
        self.wires = wires or []                # type: Iterable[Connection]
        # self.parameter_map = parameters or []   # type: List[ParameterPairing]
        # self.parameter_map = VariableMap(self, 'parameters')
        # self.state_map = VariableMap(self, 'state')
        # self.wiring_map = WiringMap(self)
        # self.__system_model = None

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


    # def initial_state(self, parameters):
    #     state = []
    #     inverse_map = self.parameter_map.invert(parameters)
    #     for component in self.components:
    #         try:
    #             component_params = inverse_map[component]
    #             state += component.initial_state(component_params)
    #         except (KeyError, TypeError):
    #             pass
    #     return state

    # @property
    # def signature(self):
    #     inputs, outputs, constraints = self.wiring_map.size()
    #     return Signature(
    #         inputs=inputs,
    #         outputs=outputs,
    #         state=len(self.state_map),
    #         parameters=len(self.parameter_map),
    #         constraints=constraints
    #     )
    #
    # def compute_dynamics(self, t, state, algebraic, inputs, parameters):
    #     states = self.state_map.invert(state)
    #     params = self.parameter_map.invert(parameters)
    #     input_vars, algebraic_vars = self.wiring_map.invert_inputs(
    #         inputs, algebraic
    #     )
    #
    #     dxdt = [0] * self.signature.state
    #
    #     for component in self.components:
    #         if component.signature.state == 0:
    #             continue
    #         try:
    #             this_state = states[component]
    #             these_params = params[component]
    #             these_inputs = input_vars[component]
    #             these_algebraics = algebraic_vars[component]
    #
    #             dxdt += component.compute_dynamics(
    #                 t, this_state, these_algebraics, these_inputs, these_params
    #             )
    #
    #         except AttributeError:
    #             pass
    #     return dxdt
    #
    # def compute_outputs(self, t, state, algebraic, inputs, parameters):
    #     outputs, _ = self._compute_combined_residuals(
    #         t, state, algebraic, inputs, parameters
    #     )
    #     return outputs
    #
    # def compute_residuals(self, t, state, algebraic, inputs, parameters):
    #     _, residuals = self._compute_combined_residuals(
    #         t, state, algebraic, inputs, parameters
    #     )
    #     return residuals
    #
    # def _compute_combined_residuals(
    #         self, t, state, algebraic, inputs, parameters):
    #     states = self.state_map.invert(state)
    #     params = self.parameter_map.invert(parameters)
    #     input_vars, algebraic_vars = self.wiring_map.invert_inputs(
    #         inputs, algebraic
    #     )
    #     _, n_outputs, n_constraints = self.wiring_map.size()
    #     outputs = [0] * n_outputs
    #     residuals = [0] * n_constraints
    #
    #     for component in self.components:
    #         args = (
    #             t,
    #             states[component],
    #             algebraic_vars[component],
    #             input_vars[component],
    #             params[component]
    #         )
    #         try:
    #             these_outputs = component.compute_outputs(*args)
    #             self.wiring_map.apply_outputs(
    #                 component, these_outputs, outputs, residuals
    #             )
    #         except AttributeError:
    #             pass
    #
    #         try:
    #             these_residuals = component.compute_residuals(*args)
    #             self.wiring_map.apply_residuals(
    #                 component, these_residuals, residuals
    #             )
    #         except AttributeError:
    #             pass
    #
    #     return outputs, residuals

    # def generate_expressions(self, context):
    #     f, g, h = zip(*[component.generate_expressions(context)
    #                   for component in self.components])
    #
    #     for outs, ins in self.wires:
    #         u = context.get_or_create_symbol(ins)
    #         y = context.get_or_create_symbol(outs)
    #         h.append(u == y)
    #
    #     return f, g, h

