from __future__ import annotations

from collections import Counter
from math import floor
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
import csv
from typing import List, Sequence, Tuple, NamedTuple, overload
import numpy as np
from flowpic_dataset.processors import QuickFlowFileProcessor, QuickPcapFileProcessor, BasicProcessor
from misc.data_classes import Flow, Block
from misc.utils import build_pic


class FlowDataSet(Dataset):
    def __init__(self, data: Sequence[Sequence[Tuple[float, int]]], labels: Sequence[int],
                 start_times: Sequence[float] = None,
                 transform=build_pic):
        """

        :param data: a sequence of blocks where each block contains a sequence of packets ([size, arrival_time])
        :param labels: the labeling of the blocks in data
        :param start_times: the start times of the blocks in data
        :param transform: applied on the block when calling __getitem__, by default creates the FlowPic
        """
        if start_times is None:
            start_times = [0.0] * len(data)

        self._data = np.array(data, dtype=list)
        self.labels = np.array(labels)
        self.start_times = np.array(start_times)
        self.transform = transform

    @classmethod
    def from_blocks_file(cls, csv_file_path, global_label=0):
        with open(csv_file_path, newline='') as f:
            start_times, _, data = zip(*[Block.create(row) for row in csv.reader(f, delimiter=',')])
            labels = np.array([global_label] * len(data))

            return FlowDataSet(data, labels, start_times)

    @classmethod
    def from_flows_file(cls, csv_file_path, global_label=0,
                        block_duration_in_seconds: int = 60,
                        block_delta_in_seconds: int = 15,
                        packet_size_limit: int = 1500
                        ):
        p = QuickFlowFileProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        blocks = p.transform_file_to_blocks(csv_file_path)
        return cls.from_blocks(blocks, global_label)

    @classmethod
    def from_flows(cls, flows: Sequence[Flow], global_label=0,
                   block_duration_in_seconds: int = 60,
                   block_delta_in_seconds: int = 15,
                   packet_size_limit: int = 1500
                   ):
        p = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        blocks = p.split_multiple_flows_to_blocks(flows)
        return cls.from_blocks(blocks, global_label)

    @classmethod
    def from_blocks(cls, blocks: Sequence[Block], global_label=0):
        start_times, _, data = zip(*blocks)
        labels = np.array([global_label] * len(data))

        return FlowDataSet(data, labels, start_times)

    def get_block(self, index: int) -> Block:
        stream = self._data[index]
        return Block(self.start_times[index], len(stream), stream)

    def get_blocks(self, indices: Sequence[int]) -> Sequence[Block]:
        return [self.get_block(i) for i in indices]

    def get_num_classes(self):
        return len(set(self.labels))

    @staticmethod
    def concatenate(datasets: Sequence[FlowDataSet]) -> FlowDataSet:
        return sum(datasets[1:], datasets[0])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        x = self._data[idx]
        if self.transform:
            x = self.transform(x).unsqueeze(0)

        return x, np.long(self.labels[idx])

    def __add__(self, other):
        if not isinstance(other, FlowDataSet):
            raise Exception("other is not of type FlowsDataSet")

        self._data = np.concatenate([self._data, other._data])
        self.labels = np.concatenate([self.labels, other.labels])
        self.start_times = np.concatenate([self.start_times, other.start_times])

        return self

    def __str__(self) -> str:
        return 'FlowDataSet:\n' + '   num samples: ' + str(len(self._data)) + '\n   label count: ' + \
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
