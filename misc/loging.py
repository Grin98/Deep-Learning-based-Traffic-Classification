import sys


class Logger:
    def __init__(self, file=None, verbose: bool = True):
        if file is None:
            file = sys.stdout
        self.file = file
        self.verbose = verbose

    def write(self, *values):
        print(*values, file=self.file)

    def write_verbose(self, *values):
        if self.verbose:
            self.write(*values)
