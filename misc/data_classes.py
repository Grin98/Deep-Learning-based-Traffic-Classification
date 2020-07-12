from __future__ import annotations
from typing import NamedTuple, List, Tuple, Sequence, Any
import numpy as np

from misc.constants import PACKET_SIZE_LIMIT


class Flow(NamedTuple):
    app: str
    five_tuple: List[str]
    start_time: float
    num_packets: int
    times: np.ndarray
    sizes: np.ndarray

    @classmethod
    def create_from_row(cls, row: List[str]):
        app = row[0]
        five_tuple = row[1:6]
        start_time = float(row[6])
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes
        return cls.create(app, five_tuple, start_time, times, sizes)

    @classmethod
    def create(cls, app: str,
               five_tuple: Sequence,
               start_time: float,
               times: Sequence,
               sizes: Sequence,
               normalize: bool = False
               ):
        times = np.array(times, dtype=float)
        sizes = np.array(sizes, dtype=int)

        mask = sizes <= PACKET_SIZE_LIMIT
        times = times[mask]
        sizes = sizes[mask] - 1
        num_packets = len(times)
        if normalize:
            times -= start_time

        return Flow(app, list(five_tuple), start_time, num_packets, times, sizes)

    def convert_to_row(self):
        """
        :return: returns the flow as a list in a format for saving it in a csv file
        """
        # self.sizes + 1 because when flow is created it subtracts 1 from sizes
        row = [self.app] + self.five_tuple + [self.start_time, self.num_packets] + \
              list(self.times) + [' '] + list(self.sizes + 1)
        return row

    @staticmethod
    def change_app(flow, app: str) -> Flow:
        return Flow.create(app, flow.five_tuple, flow.start_time, flow.times, flow.sizes)

    @staticmethod
    def change_start_time(flow, start_time: float) -> Flow:
        return Flow.create(flow.app, flow.five_tuple, start_time, flow.times, flow.sizes)


class Block(NamedTuple):
    start_time: float
    num_packets: int
    times: np.array
    sizes: np.array

    @classmethod
    def create_from_row(cls, row: List[str]):
        start_time = float(row[0])
        num_packets = int(row[1])
        off_set = 2  # meta data occupies first indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set):]

        # casting from string
        times = np.array(times, dtype=float)
        sizes = np.array(sizes, dtype=int)

        return Block(start_time, num_packets, times, sizes)

    @classmethod
    def create_from_stream(cls, start_time: float, data: List[Tuple[float, int]]):
        """
        This method meant to be used on data from a stream and not from Flow (where it has already been processed).
        Mainly used in live_capture.

        :param start_time: the start time of the block
        :param data: a list of tuples where the first is the arriavle time of a packet and the second one is the
                packet size
        :param normalize: in case the time of the packets is not between 0 and BLOCK_DURATION the start_time will
                be subtracted from all the time values of the packets in data
        :return: a new Block object
        """
        times, sizes = zip(*data)
        times = np.array(times, dtype=float)
        sizes = np.array(sizes, dtype=int)

        mask = sizes <= PACKET_SIZE_LIMIT
        times = times[mask]
        sizes = sizes[mask] - 1
        num_packets = len(times)
        times -= start_time

        return Block(start_time, num_packets, times, sizes)

    def convert_to_row(self):
        """
        :return: returns the block as a list in a format for saving it in a csv file
        """
        row = [self.start_time, self.num_packets] + self.times.tolist() + self.sizes.tolist()
        return row


class ClassifiedFlow(NamedTuple):
    flow: Flow
    pred: int
    classified_blocks: Sequence[ClassifiedBlock]


class ClassifiedBlock(NamedTuple):
    block: Block
    pred: int
    probabilities: Sequence[float]


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
