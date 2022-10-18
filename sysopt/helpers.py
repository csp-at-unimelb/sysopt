"""Common helper functions."""
from typing import List, Iterable, Type, Iterator


def flatten(the_list: List, depth: int = 1) -> List:
    """Recursively flatten the list to the given depth."""

    result = []
    for item in the_list:
        if depth > 0 and isinstance(item, (list, tuple)):
            result += flatten(item, depth - 1)
        else:
            result.append(item)

    return result


def strip_nones(the_list: List) -> List:
    """Returns a copy of the list with all Nones removed."""
    return [l_i for l_i in the_list if l_i is not None]


def filter_by_class(iterable: Iterable, cls: Type) -> Iterator:
    """Gets an iterator that returns each instance of the given class."""
    for item in iterable:
        if isinstance(item, cls):
            yield item


def slice_to_list(slce: slice, max_len=None):
    n = slce.stop or max_len
    return list(range(n))[slce]
