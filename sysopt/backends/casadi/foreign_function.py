import casadi
from typing import List, Tuple


from sysopt.symbolic.symbols import Function, Algebraic, Compose, SymbolicArray
from sysopt.backends.casadi.compiler import implements
from sysopt.backends.casadi.expression_graph import substitute

__all__ = []


class CasadiJacobian(casadi.Callback):
    def __init__(self,
                 name,
                 func,
                 f_arguments: List[SymbolicArray],
                 f_shape: Tuple[int],
                 opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        n = 0
        self.arg_offsets = [0]
        for arg in f_arguments:
            n += len(arg)
            self.arg_offsets.append(n)
        m, = f_shape
        self._shape = (n, m)

        self.construct(name, opts)

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        return casadi.Sparsity.dense(self._shape[idx], 1)

    def get_sparsity_out(self, idx):
        return casadi.Sparsity.dense(*self._shape)

    def eval(self, arg):
        x_vec = arg[0]
        x_arguments = [
            x_vec[i:i_next]
            for i, i_next in zip(self.arg_offsets[:-1], self.arg_offsets[1:])
        ]

        result = self.func(*x_arguments)
        assert len(result) == len(x_arguments),\
            'Jacobian must return a matrix for each vector arguments'
        results = [casadi.DM(r).T for r in result]
        jacobian = casadi.vertcat(*results)
        return [jacobian]


class CasadiFFI(casadi.Callback):
    def __init__(self, function, arguments, shape, jacobian=None,
                 references=None,
                 name='f',
                 opts={}):
        casadi.Callback.__init__(self)
        self.references = references or None
        self.func = function
        self._offsets = [0]
        self.arguments = arguments
        self._shape = shape
        for arg in arguments:
            self._offsets.append(self._offsets[-1] + len(arg))

        self._jacobian = None
        self._outs = shape[0]
        self._jacobian_impl = None
        if jacobian is not None:
            self._jacobian = jacobian
        self.construct(name, opts)

    def has_jacobian(self):
        result = self._jacobian is not None
        return result

    def get_jacobian(self, name, *args, **kwargs):
        self._jacobian_impl = CasadiJacobian(
            name, self._jacobian, self._arguments, self._shape
        )
        return self._jacobian_impl

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        return casadi.Sparsity.dense(self._offsets[-1], 1)

    def get_sparsity_out(self, idx):
        return casadi.Sparsity.dense(self._outs, 1)

    def eval(self, args):
        arg = args[0]
        inner_args = [
            arg[i: i_next]
            for i, i_next in zip(self._offsets[:-1], self._offsets[1:])
        ]

        result = self.func(*inner_args)

        return [casadi.vertcat(*result)]


@implements(Function)
def wrap_function(func: Function):

    impl = CasadiFFI(
        func.function, func.arguments, func.shape, jacobian=func.jacobian
    )
    shape = func.shape
    args = func.arguments

    return CasadiForeignFunction(impl, args, shape)


class CasadiForeignFunction(Algebraic):
    def __init__(self, impl, arguments, shape, name='f', refs=None):
        self.arguments = arguments
        self._shape = shape
        self.impl = impl
        self.name = name
        self.refs = refs or []

    def symbols(self):
        return set(self.arguments)

    @property
    def shape(self):
        return self._shape

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'Casadi implementation of {self.name}'

    def __call__(self, *args):
        x = casadi.vertcat(*args)

        result = self.impl(x)
        try:
            return result.full()
        except AttributeError:
            return result

    def call(self, arg_dict):
        args = [arg_dict[a] for a in self.arguments]
        return self.__call__(*args)


@implements(Compose)
def compose_implementation(composition: Compose):
    f = wrap_function(composition.function)

    outer_args = {
        arg: casadi.MX.sym(str(arg), len(arg)) for arg in composition.arguments
    }

    inner_args = {
        k: substitute(v, outer_args)
        for k, v in composition.arg_map.items()
    }
    args = [inner_args[a] for a in composition.function.arguments]

    f_of_g = f(*args)
    x = casadi.vertcat(*list(outer_args.values()))
    impl = casadi.Function('composition',[x], [f_of_g])

    return CasadiForeignFunction(
        impl=impl,
        arguments=composition.arguments,
        shape=composition.function.shape,
        refs=[f, f_of_g]
    )



