from __future__ import annotations

from collections import Counter
from pathlib import Path
from torch.utils.data.dataset import Dataset
import csv
from typing import List, Sequence, Tuple, NamedTuple
import numpy as np
from flowpic_dataset.flowpic_builder import FlowPicBuilder
from flowpic_dataset.processors import QuickFlowFileProcessor, QuickPcapFileProcessor, BasicProcessor


class Block(NamedTuple):
    data: Sequence[Tuple[int, float]]
    start_time: float


class FlowsDataSet(Dataset):
    def __init__(self, data: Sequence[Sequence[Tuple[int, float]]], labels: Sequence[int],
                 start_times: Sequence[float] = None,
                 transform=FlowPicBuilder().build_pic):
        """

        :param data: a sequence of blocks where each block contains a sequence of packets ([size, arrival_time])
        :param labels: the labeling of the blocks in data
        :param start_times: the start times of the blocks in data
        :param transform: applied on the block when calling __getitem__, by default creates the FlowPic
        """
        if start_times is None:
            start_times = [0.0] * len(data)

        self.data = np.array(data)
        self.labels = np.array(labels)
        self.start_times = np.array(start_times)
        self.transform = transform

    @classmethod
    def from_blocks_file(cls, csv_file_path, global_label=0):
        with open(csv_file_path, newline='') as f:
            data, start_times = zip(*[cls.transform_row_to_block(row) for row in csv.reader(f, delimiter=',')])
            labels = np.array([global_label] * len(data))

            return FlowsDataSet(data, labels, start_times)

    @classmethod
    def from_flows_file(cls, csv_file_path, global_label=0,
                        block_duration_in_seconds: int = 60,
                        block_delta_in_seconds: int = 15,
                        packet_size_limit: int = 1500):
        p = QuickFlowFileProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        block_rows = p.transform_file_to_block_rows(csv_file_path)
        return cls.from_block_rows(block_rows, global_label)

    @classmethod
    def from_flow_rows(cls, flow_rows, global_label=0,
                       block_duration_in_seconds: int = 60,
                       block_delta_in_seconds: int = 15,
                       packet_size_limit: int = 1500):

        p = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        flows = [p.transform_row_to_flow(row) for row in flow_rows]
        block_rows = p.split_multiple_flows_to_block_rows(flows)
        return cls.from_block_rows(block_rows, global_label)

    @classmethod
    def from_block_rows(cls, block_rows, global_label=0):
        data, start_times = zip(*[cls.transform_row_to_block(row) for row in block_rows])
        labels = np.array([global_label] * len(data))

        return FlowsDataSet(data, labels, start_times)

    @staticmethod
    def concatenate(datasets: Sequence[FlowsDataSet]) -> FlowsDataSet:
        return sum(datasets[1:], datasets[0])

    @staticmethod
    def transform_row_to_block(row: List[str]) -> Block:
        start_time = float(row[0])
        num_packets = int(row[1])
        off_set = 2  # meta data occupies first indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set):]

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        return Block(data=list(zip(sizes, times)), start_time=start_time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x).unsqueeze(0)

        return x, np.long(self.labels[idx])

    def __add__(self, other):
        if not isinstance(other, FlowsDataSet):
            raise Exception("other is not of type FlowsDataSet")

        self.data = np.concatenate([self.data, other.data])
        self.labels = np.concatenate([self.labels, other.labels])
        self.start_times = np.concatenate([self.start_times, other.start_times])

        return self

    def __str__(self) -> str:
        return 'FlowDataSet:\n' + '   num samples: ' + str(len(self.data)) + '\n   label count: ' + \
               str(Counter(self.labels))

    # def split_set(self, train_percent: float) -> Tuple[FlowsDataSet, FlowsDataSet]:
    #     test_size = int(self.__len__() * (1 - train_percent))
    #     indices = random.sample(range(self.__len__()), test_size)
    #
    #     test_data = self.data[indices]
    #     test_labels = self.labels[indices]
    #
    #     train_data = np.delete(self.data, indices)
    #     train_labels = np.delete(self.labels, indices)
    #
    #     return FlowsDataSet(data=train_data, labels=train_labels), FlowsDataSet(data=test_data, labels=test_labels)
    #
    # def balance(self, num_samples_per_class: int):
    #
    #     label_count = list(self._get_label_count().items())
    #     under_sample_quantities = dict(
    #         map(lambda key_val: (key_val[0], min(key_val[1], num_samples_per_class)), label_count)
    #     )
    #
    #     over_sample_quantities = dict(
    #         map(lambda key_val: (key_val[0], max(key_val[1], num_samples_per_class)), under_sample_quantities.items())
    #     )
    #
    #     rus = RandomUnderSampler(under_sample_quantities, replacement=False)
    #     ros = RandomOverSampler(over_sample_quantities)
    #
    #     self._apply_sampler(rus)
    #     self._apply_sampler(ros)
    #
    # def _apply_sampler(self, sampler):
    #     dummy_data = [[0]] * len(self.data)
    #     sampler.fit_resample(dummy_data, self.labels)
    #     indices = sampler.sample_indices_
    #
    #     self.data = self.data[indices]
    #     self.labels = self.labels[indices]
    #
    # def create_weighted_random_sampler(self, num_to_sample: int, replacement: bool = False) -> WeightedRandomSampler:
    #     return WeightedRandomSampler(self._create_dataset_weights(),
    #                                  num_to_sample,
    #                                  replacement=replacement)
    #
    # def _create_dataset_weights(self) -> List[float]:
    #     label_probabilities = list(map(lambda x: 1 / x,
    #                                    self._get_label_count().values()
    #                                    ))
    #     return [label_probabilities[y] for y in self.labels]
    #
    # def _get_label_count(self):
    #     return dict(sorted(Counter(self.labels).items(),
    #                        key=lambda item: item[0]))
