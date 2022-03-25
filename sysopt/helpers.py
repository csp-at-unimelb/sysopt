"""Common helper functions."""


def flatten(the_list, depth=1):
    result = []
    for item in the_list:
        if depth > 0 and isinstance(item, (list, tuple)):
            result += flatten(item, depth - 1)
        else:
            result.append(item)

    return result


def strip_nones(the_list):
    return [l_i for l_i in the_list if l_i is not None]


def filter_by_class(iterable, cls):
    for item in iterable:
        if isinstance(item, cls):
            yield item
