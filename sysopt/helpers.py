def flatten(the_list):
    return [i for sublist in the_list for i in sublist]


def filter_by_class(iterable, cls):
    for item in iterable:
        if isinstance(item, cls):
            yield item
