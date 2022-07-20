import casadi


from sysopt.symbolic.symbols import Function, Algebraic, Compose
from sysopt.backends.casadi.compiler import implements
from sysopt.backends.casadi.expression_graph import substitute

__all__ = []


class CasadiFFI(casadi.Callback):
    def __init__(self, function, arguments, shape, jacobian=None):
        casadi.Callback.__init__(self)
        self.func = function
        self.sparsity_in = [
            casadi.Sparsity.dense(len(arg), 1) for arg in arguments
        ]
        self.sparsity_out = [casadi.Sparsity.dense(shape[0], 1)]

        self.construct('f', {})

    def get_n_in(self):
        return len(self.sparsity_in)

    def get_n_out(self):
        return len(self.sparsity_out)

    def get_sparsity_in(self, idx):
        return self.sparsity_in[idx]

    def get_sparsity_out(self, idx):
        return self.sparsity_out[idx]

    def eval(self, args):
        inner_args = [casadi.vertsplit(a) if a.shape[0] > 1 else a
                      for a in args]
        result = self.func(*inner_args)

        return [casadi.vertcat(*result)]


@implements(Function)
def wrap_function(func: Function):
    return CasadiForeignFunction(func)


class CasadiForeignFunction(Algebraic):
    def __init__(self, func: Function):
        self.func = func

        self.impl = CasadiFFI(func.function, func.arguments, func.shape)

    def symbols(self):
        return self.func.symbols()

    @property
    def shape(self):
        return self.func.shape

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'Casadi implementation of {self.func}'

    def __call__(self, *args):

        result = self.impl(*args)
        return result.full()

    def call(self, arg_dict):
        args = [arg_dict[a] for a in self.func.arguments]
        return self.__call__(*args)


@implements(Compose)
def compose_implementation(composition: Compose):

    f = CasadiForeignFunction(composition.function)

    outer_args = {
        arg: casadi.MX.sym(str(arg), len(arg)) for arg in composition.arguments
    }

    inner_args = {
        k: substitute(v, outer_args)
        for k, v in composition.arg_map.items()
    }
    args = [inner_args[a] for a in f.symbols()]
    f_of_g = f(*args)
    print(f_of_g)

    assert False
    # print(composition.arg_map)


