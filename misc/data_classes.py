from __future__ import annotations
from typing import NamedTuple, List, Tuple, Sequence, Any
import numpy as np


class Flow(NamedTuple):
    app: str
    five_tuple: List[str]
    start_time: float
    num_packets: int
    times: np.ndarray
    sizes: np.ndarray

    @classmethod
    def create_from_row(cls, row: List[str], packet_size_limit: int):
        app = row[0]
        five_tuple = row[1:6]
        start_time = float(row[6])
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes
        return cls.create(app, five_tuple, packet_size_limit, times, sizes, start_time)

    @classmethod
    def create(cls, app: str,
               five_tuple: Sequence,
               packet_size_limit: int,
               times: Sequence,
               sizes: Sequence,
               start_time: float = None
               ):
        num_packets = len(times)
        times = np.array(times, dtype=float)
        sizes = np.array(sizes, dtype=int)

        mask = sizes <= packet_size_limit
        times = times[mask]
        sizes = sizes[mask] - 1

        if start_time is None:
            start_time = times[0]
            times -= start_time

        return Flow(app, list(five_tuple), start_time, num_packets, times, sizes)

    def convert_to_row(self):
        # self.sizes + 1 because when flow is created it subtracts 1 from sizes
        row = [self.app] + self.five_tuple + [self.start_time, self.num_packets] +\
              list(self.times) + [' '] + list(self.sizes + 1)
        return row


class Block(NamedTuple):
    start_time: float
    num_packets: int
    data: List[Tuple[float, int]]

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

        return Block(start_time, num_packets, list(zip(times, sizes)))

    def convert_to_row(self):
        times, sizes = zip(*self.data)
        row = [self.start_time, self.num_packets] + list(times) + list(sizes)
        return row


class ClassifiedFlow(NamedTuple):
    flow: Flow
    pred: int
    classified_blocks: Sequence[ClassifiedBlock]


class ClassifiedBlock(NamedTuple):
    block: Block
    pred: int


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_total: int
    num_correct: int
    f1_score: float
    f1_per_class: List[float]


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float
    f1: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    train_f1: List[float]
    test_loss: List[float]
    test_acc: List[float]
    test_f1: List[float]

