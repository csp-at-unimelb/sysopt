"""Decorators to ensure tensor operations have the right shapes."""


def require_equal_domains(func):
    def validator(a, b):
        if a.domain != b.domain:
            msg = f'Domains {a.domain} != {b.domain} for arguments {a}, {b}'
            raise TypeError(msg)
        return func(a, b)
    return validator


def require_equal_order_codomain(func):
    def validator(a, b):
        if ((isinstance(a.codomain, int) and isinstance(b.codomain, int))
                or len(a.codomain) == len(b.codomain)):
            return func(a, b)
        else:
            msg = f'Codomains {a.codomain}, {b.codomain} are not compatible'\
                  f'for arguments {a}, {b}'
            raise TypeError(msg)

    return validator
