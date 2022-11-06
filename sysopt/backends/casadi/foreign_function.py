"""Casadi Implementation of foreign function interfaces."""

import casadi
from typing import List, Tuple, Callable, Optional

from sysopt.symbolic.core import Function, Algebraic, Composition, SymbolicArray
from sysopt.backends.casadi.expression_graph import substitute
from sysopt.backends.implementation_hooks import get_backend

__all__ = []
__functions = []

backend = get_backend('casadi')


class CasadiJacobian(casadi.Callback):
    """Wrapper for numerical jacobians

    Args:
        name:   Name of the jacobian
        func:   Function to compute the jacobians with respect to each input
        f_arguments: Arguments for the original function
        f_shape: output shape of the original function (must be `(n,)`
                where `n` is the dimension of the codomain
        opts: Casadi-specific options

    """
    # pylint: disable=dangerous-default-value
    # as per casadi reference implementations
    def __init__(self,
                 name: str,
                 func: Callable,
                 f_arguments: List[SymbolicArray],
                 f_shape: Tuple[int],
                 hessian: Optional[Callable] = None,
                 opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        n = 0
        self.arg_offsets = [0]
        self.arg_lengths = []
        for arg in f_arguments:
            n += len(arg)
            self.arg_lengths.append(len(arg))
            self.arg_offsets.append(n)
        m, = f_shape
        self._shape = (m, n)
        jac_opts = opts.copy()
        jac_opts.update(
            dict(enable_fd=True)
        )
        self.construct(name, jac_opts)

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, idx):
        if idx == 0:
            return casadi.Sparsity.dense(self._shape[1], 1)
        else:
            return casadi.Sparsity.dense(self._shape[0], 1)

    def get_sparsity_out(self, idx):
        return casadi.Sparsity.dense(*self._shape)

    def eval(self, arg):

        x_vec = arg[0].full()
        x_arguments = [
            x_vec[i:i_next] if i_next > i + 1 else x_vec[i]
            for i, i_next in zip(self.arg_offsets[:-1], self.arg_offsets[1:])
        ]

        result = self.func(*x_arguments)

        assert len(result) == len(x_arguments),\
            'Jacobian must return a matrix for each vector arguments'
        results = [casadi.DM(r).reshape((self._shape[0], cols))
                   for r, cols in zip(result, self.arg_lengths)]

        jacobian = casadi.horzcat(*results)
        return [jacobian]


class CasadiForwards(casadi.Callback):
    """Wrapper for numerical computation of forwards derivatives.

    Args:
        name: name of the function
        func: Computes the variation, with inputs being a vector of arguments
            and a vector of corresponding rates.
        f_arguments: Symbolic arguments of the original function
        f_shape: Output shape of the original function.
        opts: Casadi-specific options.

    """

    # pylint: disable=dangerous-default-value
    # as per casadi reference implementations
    def __init__(self,
                 name: str,
                 func: Callable,
                 f_arguments: List[SymbolicArray],
                 f_shape: Tuple[int],
                 opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        n = 0
        self.arg_offsets = [0]
        self.arg_lengths = []
        for arg in f_arguments:
            n += len(arg)
            self.arg_lengths.append(len(arg))
            self.arg_offsets.append(n)
        m, = f_shape
        self._inshape = (n, 1)
        self._outshape = (m, 1)
        self.construct(name, opts)

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return casadi.Sparsity.dense(self._inshape)
        elif i == 1:
            return casadi.Sparsity(self._outshape)
        else:
            return casadi.Sparsity.dense(self._inshape)

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(self._outshape)

    def eval(self, args):

        x_vec = args[0].full()
        dx_vec = args[2].full()
        x_arguments = [
            x_vec[i:i_next] if i_next > i + 1 else x_vec[i]
            for i, i_next in zip(self.arg_offsets[:-1], self.arg_offsets[1:])
        ]
        dx_arguments = [dx_vec[i:i_next] if i_next > i + 1 else dx_vec[i]
            for i, i_next in zip(self.arg_offsets[:-1], self.arg_offsets[1:])
        ]
        result = self.func(*x_arguments, *dx_arguments)

        return [casadi.reshape(casadi.DM(result), self._outshape)]


class CasadiFFI(casadi.Callback):
    """ Wrapper for a differentiable foreign function.

    Args:
        function: The function to wrap.
        arguments: List of symbolic arguments of the correct dimension.
        shape: Output shape of the function.
        jacobian: Function to compute the jacobian with respect to each
            input variable.
        name: Name of the function
        opts: Casadi specific options.

    """
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 function: Callable,
                 arguments: List[SymbolicArray],
                 shape: Tuple[int],
                 jacobian: Optional[Callable] = None,
                 forwards: Optional[Callable] = None,
                 hessian: Optional[Callable] = None,
                 name: str = 'f',
                 opts={}):
        casadi.Callback.__init__(self)
        self.func = function
        self._offsets = [0]
        self.arguments = arguments
        self._shape = shape
        for arg in arguments:
            self._offsets.append(self._offsets[-1] + len(arg))

        self._jacobian = jacobian
        self._forwards = forwards
        self._outs = shape[0]
        self._jacobian_impl = None
        self._forwards_impl = None
        self._hessian = None
        self.construct(name, opts)

    def has_forward(self, nfwd):
        return (nfwd == 1) and self._forwards is not None

    def get_forward(self, nfwds, name,iname, oname, opts):
        self._forwards_impl = CasadiForwards(
            name, self._forwards, self.arguments, self._shape
        )
        return self._forwards_impl

    def has_jacobian(self):
        result = self._jacobian is not None
        return result

    def get_jacobian(self, name, *args, **kwargs):
        self._jacobian_impl = CasadiJacobian(
            name, self._jacobian, self.arguments, self._shape,
            self._hessian
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
        arg = args[0].full()
        inner_args = [
            arg[i: i_next] if i_next > i + 1 else arg[i]
            for i, i_next in zip(self._offsets[:-1], self._offsets[1:])
        ]

        result = self.func(*inner_args)

        return [casadi.vertcat(*result)]


@backend.implements(Function)
def wrap_function(func: Function):

    impl = CasadiFFI(
        func.function, func.arguments, func.shape,
        jacobian=func.jacobian,
        forwards=func.forwards
    )
    shape = func.shape
    args = func.arguments

    __functions.append(impl)
    return CasadiForeignFunction(impl, args, shape)


class CasadiForeignFunction(Algebraic):
    """Expression-graph compatible wrapper for a casadi-implemented functions.
    Args:
        impl: Casadi function implementation
        arguments: List of vector arguments to this function.
        shape: Output dimension of the form `(d,)`
        name: Name of this function.

    """

    def __init__(self,
                 impl: casadi.Function,
                 arguments: List[SymbolicArray],
                 shape: Tuple[int],
                 name: str = 'f'):
        self.arguments = arguments
        self._shape = shape
        self.impl = impl
        self.name = name

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
        return self(*args)


@backend.implements(Composition)
def compose_implementation(composition: Composition):
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
    impl = casadi.Function('composition', [x], [f_of_g])

    return CasadiForeignFunction(
        impl=impl,
        arguments=composition.arguments,
        shape=composition.function.shape
    )



