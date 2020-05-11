from os import listdir
from os.path import isdir, join
from pathlib import Path

from torch.utils.data.dataset import ConcatDataset

from enum import Enum
from flowpic_dataset.dataset import FlowsDataSet
from flowpic_dataset.flowpic_builder import FlowPicBuilder
import numpy as np
import torch


class Label(Enum):
    Encryption = 0
    Class = 1


class FlowPicDataLoader:
    def __init__(self, dataset_root_dir, testing=False):
        self.root_dir = Path(dataset_root_dir)
        self.testing = testing
        self.labels = {}

    def load_dataset(self, is_split: bool = False):
        """
        creates a FlowDataset from each file at the leaf level of the directory tree
        and returns a ConcatDataset of all of them
        """

        self.labels = {}
        print("=== Loading dataset from %s ===" % self.root_dir)
        if is_split:
            train_datasets = self._gather_datasets(self.root_dir/'train')
            test_dataset = self._gather_datasets(self.root_dir/'test')
            res = FlowsDataSet.concatenate(train_datasets), FlowsDataSet.concatenate(test_dataset)
        else:
            datasets = self._gather_datasets(self.root_dir)
            res = FlowsDataSet.concatenate(datasets)
        print("=== Dataset loading completed :D ===\n")

        return res

    def _gather_datasets(self, path: Path, label_level: bool = True, label: int = None):
        dirs = self._get_dir_directories(path)
        if not dirs:
            dss = [FlowsDataSet(file, label, testing=self.testing) for file in self._get_dir_csvs(path)]
            num_blocks = sum(map(len, dss))
            print("path: %s, num blocks: %d, label: %d" % (path, num_blocks, label))

            return dss

        datasets = []
        for d in dirs:
            if label_level:
                dir_name = d.parts[-1]
                self._add_directory_label(dir_name)
                label = self._get_directory_label(dir_name)

            datasets += self._gather_datasets(d, False, label)

        return datasets

    @staticmethod
    def _get_dir_items(dir_path: Path):
        return list(dir_path.glob('*'))

    def _get_dir_directories(self, dir_path: Path):
        return [d for d in self._get_dir_items(dir_path) if d.is_dir()]

    @staticmethod
    def _get_dir_csvs(dir_path: Path):
        return list(dir_path.glob('*.csv'))

    def _add_directory_label(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def _get_directory_label(self, d):
        return self.labels[d]
