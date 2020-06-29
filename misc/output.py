import sys
from typing import Sequence, List


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


class Progress:
    def __init__(self):
        self._progress_title = ''
        self._progress_sub_title = ''
        self._counter_title = ''
        self._counter = 0

    def progress_title(self, title: str):
        self._progress_title = title
        return self

    def progress_sub_title(self, title: str):
        self._progress_sub_title = title
        return self

    def counter_title(self, title: str):
        self._counter_title = title
        return self

    def set_counter(self, x: int):
        self._counter = x

    def get(self):
        if not self._progress_title:
            raise Exception('progress must have a main title')

        progress = self._progress_title
        if self._progress_sub_title:
            progress = f'{progress}\n{self._progress_sub_title}'
        if self._counter_title:
            progress = f'{progress}\n{self._counter_title}: {self._counter}'

        return progress

    def reset(self):
        self._progress_title = ''
        self._progress_sub_title = ''
        self._counter_title = ''
        self._counter = 0

    def __str__(self):
        return self.get()

    @staticmethod
    def pad_5tuple(five_tuple: List[str]) -> str:
        total_len = sum([len(item) for item in five_tuple])
        max_len = (15 * 2) + (2 * 5) + 3
        diff = max_len - total_len
        if diff > 0:
            left = '_' * (diff // 2)
            right = '_' * (diff - (diff // 2))
            return left + str(five_tuple) + right
        return str(five_tuple)
