from collections import defaultdict
from weakref import ref

__registry = {}
__type_registry = defaultdict(set)


def get_default_name(obj):
    for template, classes in __type_registry.items():
        for cls in classes:
            if isinstance(obj, cls):
                return template
    raise NotImplementedError(f"Default name not registered "
                              f"for type {type(obj)}")


def register_default_name(name):
    def register_method(cls):
        register(cls, name)
        return cls

    return register_method


def register(cls, name):
    __type_registry[name].add(cls)


def register_or_create_name(obj, name):
    if name is not None:
        if name in __registry and __registry[name]:
            raise NameError(f"Variable with name {name} already exists")
        __registry[name] = ref(obj)
        return name

    idx = 0
    while True:
        name = get_default_name(obj) + f"{idx}"
        if name not in __registry:
            __registry[name] = ref(obj)
            return name
        idx += 1
