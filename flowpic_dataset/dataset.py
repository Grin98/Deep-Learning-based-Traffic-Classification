from __future__ import annotations

from collections import Counter
from os import listdir
from os.path import join, isdir

from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import Dataset, ConcatDataset
import csv
from typing import List, Sequence, Tuple
import numpy as np
import torch
from flowpic_dataset.flowpic_builder import FlowPicBuilder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import random


class FlowsDataSet(Dataset):
    """
    parameter label will be the label of all the data entries in the file
    """

    def __init__(self, csv_file_path=None, global_label=None, transform=FlowPicBuilder().build_pic,
                 testing: bool = False, data=None,
                 labels=None):
        self.labels = []
        self.transform = transform

        if csv_file_path is not None:
            with open(csv_file_path, newline='') as f:
                if testing:
                    self.data = [row for row in csv.reader(f, delimiter=',')]
                else:
                    self.data = [self.__transform_row_to_flow__(row) for row in csv.reader(f, delimiter=',')]

            self.data = np.array(self.data)
            self.labels = np.array([global_label] * len(self.data))

        elif data is not None and labels is not None:
            self.data = np.array(data)
            self.labels = np.array(labels)

    def split_set(self, train_percent: float) -> Tuple[FlowsDataSet, FlowsDataSet]:
        test_size = int(self.__len__() * (1 - train_percent))
        indices = random.sample(range(self.__len__()), test_size)

        test_data = self.data[indices]
        test_labels = self.labels[indices]

        train_data = np.delete(self.data, indices)
        train_labels = np.delete(self.labels, indices)

        return FlowsDataSet(data=train_data, labels=train_labels), FlowsDataSet(data=test_data, labels=test_labels)

    def balance(self, num_samples_per_class: int):

        label_count = list(self._get_label_count().items())
        under_sample_quantities = dict(
            map(lambda key_val: (key_val[0], min(key_val[1], num_samples_per_class)), label_count)
        )

        over_sample_quantities = dict(
            map(lambda key_val: (key_val[0], max(key_val[1], num_samples_per_class)), under_sample_quantities.items())
        )

        rus = RandomUnderSampler(under_sample_quantities, replacement=False)
        ros = RandomOverSampler(over_sample_quantities)

        self._apply_sampler(rus)
        self._apply_sampler(ros)

    def _apply_sampler(self, sampler):
        dummy_data = [[0]] * len(self.data)
        sampler.fit_resample(dummy_data, self.labels)
        indices = sampler.sample_indices_

        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def create_weighted_random_sampler(self, num_to_sample: int, replacement: bool = False) -> WeightedRandomSampler:
        return WeightedRandomSampler(self._create_dataset_weights(),
                                     num_to_sample,
                                     replacement=replacement)

    def _create_dataset_weights(self) -> List[float]:
        label_probabilities = list(map(lambda x: 1 / x,
                                       self._get_label_count().values()
                                       ))
        return [label_probabilities[y] for y in self.labels]

    def _get_label_count(self):
        return dict(sorted(Counter(self.labels).items(),
                           key=lambda item: item[0]))

    @staticmethod
    def concatenate(datasets: Sequence[FlowsDataSet]) -> FlowsDataSet:
        return sum(datasets[1:], datasets[0])

    @staticmethod
    def __transform_row_to_flow__(row: List[str]):
        num_packets = int(row[0])
        off_set = 1  # meta data occupies first inced
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set):]

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        return list(zip(sizes, times))

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

        return self

    def __str__(self) -> str:
        return 'FlowDataSet:\n' + '   num samples: ' + str(len(self.data)) + '\n   label count: ' + \
               str(self._get_label_count())
