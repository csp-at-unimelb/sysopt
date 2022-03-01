import time


class TimerException(Exception):
    pass


class Timer:
    def __init__(self, name=None, quiet=False):
        self.name = name
        self.tstart = None
        self.quiet = quiet

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.quiet:
            if self.name:
                print(f'[{self.name}]')
            print(f'Elapsed {self.elapsed()}')

    def start(self):
        self.tstart = time.time()

    def elapsed(self):
        if not self.tstart:
            raise TimerException("Timer has not been started")

        return time.time() - self.tstart
