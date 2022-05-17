"""Operation registry for casting between different symbol types."""

_registry = {}


def register(from_type, to_type):
    def decorator(the_caster):
        keys = from_type if isinstance(from_type, (list, tuple)) \
            else [from_type]
        for t in keys:

            if t not in _registry:
                _registry[t] = {}
            else:
                assert to_type not in _registry[from_type], \
                    f'Cast from {t} to {to_type} is already defined'
            _registry[t][to_type] = the_caster

        return the_caster
    return decorator


def cast_type(var, to_type=None):
    from_type = type(var)
    if to_type is None:
        try:
            casters = _registry[from_type]
        except KeyError as ex:
            msg = f'Don\'t know how to cast {from_type}.'
            raise NotImplementedError(msg) from ex

        if len(casters) != 1:
            msg = f'Don\'t know how to uniquely cast from {from_type}'
            raise TypeError(msg)

        caster, = casters.values()
    elif to_type is from_type:
        return var
    else:
        try:
            caster = _registry[from_type][to_type]
        except KeyError as ex:
            msg = f'Don\'t know how to cast from {from_type} to {to_type}'
            raise NotImplementedError(msg) from ex

    return caster(var)


def cast_like(var, var_like):
    to_type = type(var_like)
    return cast_type(var, to_type)


