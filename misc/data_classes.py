from typing import NamedTuple, List

import numpy as np


class Flow(NamedTuple):
    sizes: np.ndarray
    times: np.ndarray
    app: str

    @classmethod
    def create(cls, row: List[str], packet_size_limit: int):
        app = row[0]
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        mask = sizes <= packet_size_limit
        times = times[mask]
        sizes = sizes[mask] - 1

        return Flow(sizes, times, app)


class BlockRow(NamedTuple):
    start_time: float
    num_packets: int
    times: List[float]
    sizes: List[int]

    @classmethod
    def create(cls, row: List[str]):
        start_time = float(row[0])
        num_packets = int(row[1])
        off_set = 2  # meta data occupies first indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set):]

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        return BlockRow(start_time, num_packets, list(times), list(sizes))

    def __iter__(self):
        row = [self.start_time, self.num_packets] + self.times + self.sizes
        return iter(row)


class Block(NamedTuple):
    start_time: float
    data: np.ndarray

    @classmethod
    def create(cls, br: BlockRow):
        return Block(br.start_time, np.array(zip(br.sizes, br.times)))
