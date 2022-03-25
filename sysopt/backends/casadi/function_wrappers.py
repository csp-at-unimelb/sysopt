from typing import Callable

import casadi

from sysopt.types import Domain
from casadi import SX, Function
from sysopt.backends.casadi.symbols import SymbolicVector, concatenate, cast


class BlockFunctionWrapper(Function):
    def __init__(self, domain: Domain,
                 codomain: int,
                 function: Callable,  *args):
        super().__init__(*args)

    def __new__(cls, domain: Domain, codomain: int, function: Callable):
        args = [
            SymbolicVector('t', domain.time),
            SymbolicVector('x', domain.states),
            SymbolicVector('z', domain.constraints),
            SymbolicVector('u', domain.inputs),
            SymbolicVector('p', domain.parameters)
        ]
        r = function(*args)
        assert len(r) == codomain
        f = Function('f', args, r)
        setattr(f, 'domain', domain)
        setattr(f, 'codomain', codomain)
        f.__class__ = BlockFunctionWrapper
        f.__bases__ = [Function]
        return f

    def __call__(self, *args):
        result = super().__call__(*args)
        if isinstance(result, casadi.DM):
            r = result.toarray()[:, 0]
            return list(r)
        else:
            return cast(result)

    @staticmethod
    def coproduct(function_list):
        domain = Domain()
        t = SymbolicVector('t', 1)
        x_list = []
        z_list = []
        u_list = []
        p_list = []
        r_list = []

        for f in function_list:
            x = SymbolicVector('x', f.domain.states)
            z = SymbolicVector('z', f.domain.constraints)
            u = SymbolicVector('u', f.domain.inputs)
            p = SymbolicVector('p', f.domain.parameters)
            x_list.append(x)
            z_list.append(z)
            u_list.append(u)
            p_list.append(p)
            res = f(t, x, z, u, p)
            if isinstance(res, (list, tuple)):
                r_list += [*res]
            else:
                r_list.append(res)

            domain += f.domain

        args = [
            t,
            concatenate(*x_list),
            concatenate(*z_list),
            concatenate(*u_list),
            concatenate(*p_list)
        ]
        res = concatenate(*r_list)
        f = Function('f', args, [res])
        codomain = len(res)
        setattr(f, 'domain', domain)
        setattr(f, 'codomain', codomain)
        setattr(f, '__class__', BlockFunctionWrapper)
        setattr(f, '__bases__', Function)
        return f


class SimpleFunctionWrapper(Function):
    def __new__(cls, domain: int, codomain: int, function: Callable):
        args = [SX.sym('p', domain)]
        r = function(*args)
        assert len(r) == codomain
        f = Function('f', args, r)
        setattr(f, 'domain', domain)
        setattr(f, 'codomain', codomain)
        return f

    def __call__(self, *args):
        result = super().__call__(*args)
        return SymbolicVector.from_sx(result)

    @staticmethod
    def coproduct(functions):

        p_list = [SymbolicVector('p', f.domain) for f in functions]
        results = [f(p) for f, p in zip(functions, p_list)]

        args = concatenate(*p_list)
        res = concatenate(*results)
        domain = len(args)
        codomain = len(res)
        f = Function('f', [args], [res])
        setattr(f, 'domain', domain)
        setattr(f, 'codomain', codomain)
        setattr(f, '__class__', SimpleFunctionWrapper)
        setattr(f, '__bases__', Function)
        return f