from __future__ import annotations

from collections import Counter
from math import floor
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
import csv
from typing import List, Sequence, Tuple, NamedTuple, overload
import numpy as np
from flowpic_dataset.processors import QuickFlowFileProcessor, BasicProcessor
from misc.data_classes import Flow, Block
from misc.utils import build_pic


class BlocksDataSet(Dataset):
    """
    A Dataset which contains blocks and when accessed returns the blocks as FlowPics (2D-Histograms) with
    their labels
    """
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
        with open(csv_file_path, newline='', mode='r') as f:
            start_times, _, data = zip(*[Block.create_from_row(row) for row in csv.reader(f, delimiter=',')])
            labels = np.array([global_label] * len(data))

            return BlocksDataSet(data, labels, start_times)

    @classmethod
    def from_flows_file(cls, csv_file_path, global_label=0):
        p = QuickFlowFileProcessor()
        blocks = p.transform_file_to_blocks(csv_file_path)
        return cls.from_blocks(blocks, global_label)

    @classmethod
    def from_flows(cls, flows: Sequence[Flow], global_label=0):
        p = BasicProcessor()
        blocks = p.split_multiple_flows_to_blocks(flows)
        return cls.from_blocks(blocks, global_label)

    @classmethod
    def from_blocks(cls, blocks: Sequence[Block], global_label=0):
        start_times, _, data = zip(*blocks)
        labels = np.array([global_label] * len(data))

        return BlocksDataSet(data, labels, start_times)

    def get_block(self, index: int) -> Block:
        stream = self._data[index]
        return Block(self.start_times[index], len(stream), stream)

    def get_blocks(self, indices: Sequence[int]) -> Sequence[Block]:
        return [self.get_block(i) for i in indices]

    def get_num_classes(self):
        return len(set(self.labels))

    @staticmethod
    def concatenate(datasets: Sequence[BlocksDataSet]) -> BlocksDataSet:
        """

        :param datasets: a sequence of BlockDatasets
        :return: merges all datasets into the first one and returns it
        """
        return sum(datasets[1:], datasets[0])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        x = self._data[idx]
        if self.transform:
            x = self.transform(x).unsqueeze(0)
            # x = np.expand_dims(self.transform(x), axis=0)
        return x, np.long(self.labels[idx])

    def __add__(self, other):
        if not isinstance(other, BlocksDataSet):
            raise Exception("other is not of type FlowsDataSet")

        self._data = np.concatenate([self._data, other._data])
        self.labels = np.concatenate([self.labels, other.labels])
        self.start_times = np.concatenate([self.start_times, other.start_times])

        return self

    def __str__(self) -> str:
        count = Counter(self.labels)
        count = sorted(count.items())
        count_str = ''
        for k, v in count:
            count_str += f'{k}: {v}, '

        return f'FlowDataSet:\n' \
            f'num samples: {str(len(self._data))}\n' \
            f'label count: {count_str}'
