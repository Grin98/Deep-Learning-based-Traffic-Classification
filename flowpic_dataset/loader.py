from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Tuple, Union

from torch.utils.data import DataLoader

from flowpic_dataset.dataset import BlocksDataSet

from misc.utils import get_dir_directories, get_dir_csvs


class Format(Enum):
    Default = 0
    Split = 1
    SplitCV = 2


class FlowCSVDataLoader:

    def __init__(self, testing=False, verbose: bool = True):
        self.testing = testing
        self.verbose = verbose
        self.labels = {}

    def load_cross_validation_dataset(self, dataset_root_dir, test_group_index: int) -> Tuple[BlocksDataSet, BlocksDataSet]:
        root_dir = Path(dataset_root_dir)
        dirs = get_dir_directories(root_dir)

        test_dataset = self.load_dataset(dir([test_group_index]), format_=Format.Default)
        train_datasets = [self.load_dataset(d, format_=Format.Default)
                          for i, d in enumerate(dirs)
                          if i != test_group_index]

        return BlocksDataSet.concatenate(train_datasets), test_dataset

    def load_dataset(self, dataset_root_dir, format_: Format) -> Union[Tuple[BlocksDataSet], BlocksDataSet]:
        root_dir = Path(dataset_root_dir)
        self.labels = {}

        self._print_verbose("=== Loading dataset from %s ===" % root_dir)
        if format_ == Format.Default:
            datasets = self._gather_datasets(root_dir, level=0)
            res = BlocksDataSet.concatenate(datasets)
        else:
            self._print_verbose('train')
            level = 0 if format_ == Format.Split else 1
            train_datasets = self._gather_datasets(root_dir/'train', level=level)

            self._print_verbose('test')
            test_dataset = self._gather_datasets(root_dir/'test', level=0)
            res = BlocksDataSet.concatenate(train_datasets), BlocksDataSet.concatenate(test_dataset)
        self._print_verbose("=== Dataset loading completed :D ===")

        return res

    def _gather_datasets(self, path: Path, level: int, label: int = 0):
        dirs = get_dir_directories(path)
        if not dirs:
            dss = [BlocksDataSet.from_blocks_file(file, label) for file in get_dir_csvs(path)]
            num_blocks = sum(map(len, dss))
            self._print_verbose("path: %s, num blocks: %d, label: %d" % (path, num_blocks, label))

            return dss

        labels = None
        if self.is_label_level(level):
            labels = sorted(list(map(lambda dir_: dir_.parts[-1], dirs)))

        datasets = []
        for d in dirs:
            if self.is_label_level(level):
                dir_name = d.parts[-1]
                label = labels.index(dir_name)

            datasets += self._gather_datasets(d, level-1, label)

        return datasets

    @staticmethod
    def is_label_level(level: int):
        return level == 0

    def _add_directory_label(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def _get_directory_label(self, d):
        return self.labels[d]

    def _print_verbose(self, s: str):
        if self.verbose:
            print(s)






