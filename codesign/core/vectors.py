from codesign.core.tree_base import Algebraic, compare
from codesign.core.name_registry import register_or_create_name, register_default_name
from typing import List


@register_default_name('X')
class Vector(Algebraic):
    def __init__(self, data: List, *args, name=None, **kwargs):
        self.data = tuple([l for l in data])
        self.name = register_or_create_name(self, name)

    @property
    def shape(self):
        return len(self.data),

    def __contains__(self, item):
        return any(item is d for d in self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str([repr(l) for l in self.data])

    def __hash__(self):
        return hash((hash(d) for d in self.data))

    def __len__(self):
        return len(self.data)

    def atoms(self):
        atoms = set()
        for entry in self.data:
            try:
                atoms |= entry.atoms()
            except AttributeError:
                pass
        return atoms

    def __cmp__(self, other):
        try:
            return [
                all(compare(a, b) for a, b in zip(self.data, other.data))
            ]
        except AttributeError:
            return False

    @staticmethod
    def create_filled_with(factory, size: int):
        return Vector([factory() for _ in range(size)])

    def index(self, item):
        results = [
            i for i, elem in enumerate(self.data)
            if elem is item
        ]

        return results[0] if results else None


@register_default_name('M')
class DenseArray(Algebraic):
    def __init__(self, data: List[List], *args, **kwargs):
        n = len(data)
        m = max(len(l) for l in data)
        self.shape = (n, m)
        self.data = data

    def __hash__(self):
        return hash((hash(d) for row in self.data for d in row))

    def atoms(self):
        atoms = set()
        for row in self.data:
            for cell in row:
                try:
                    atoms |= cell.atoms()
                except AttributeError:
                    pass
        return atoms


def is_scalar(arg):
    try:
        n, m = arg.shape
        return not (n > 1 or m and m > 1)
    except AttributeError:
        return True


