"""Common helper functions."""


def flatten(the_list):
    return [i for sublist in the_list for i in sublist]


def filter_by_class(iterable, cls):
    for item in iterable:
        if isinstance(item, cls):
            yield item


class InterpolatedPath:
    """Function that linearly interpolates between the data-points."""

    def __init__(self, t, x):
        self.t = t
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    @property
    def t_max(self):
        return self.t[-1]

    def apply(self, projection):
        return InterpolatedPath(self.t, projection @ self.x)

    def __call__(self, t):
        """Get the value at `t`, assumed between 0 and T_max."""
        for i in range(self.t.shape[1] - 1):
            if self.t[i] <= t < self.t[i + 1]:
                dist = self.t[i + 1] - self.t[i]
                w0 = (self.t[i + 1] - t) / dist
                w1 = (t - self.t[i]) / dist
                return w0 * self.x[:, i] + w1 * self.x[:, i + 1]

        if abs(t - self.t[-1]) < 1e-7:
            return self.x[:, -1]

        raise ValueError(f'No data point for {t}')
