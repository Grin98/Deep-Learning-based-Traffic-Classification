from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Tuple, Union

from torch.utils.data import DataLoader

from flowpic_dataset.dataset import BlocksDataSet
from misc.loging import Logger

from misc.utils import get_dir_directories, get_dir_csvs


class Format(Enum):
    Default = 0
    Split = 1
    SplitCV = 2


class FlowCSVDataLoader:

    def __init__(self, log: Logger = Logger(), testing=False):
        self.log = log
        self.testing = testing
        self.labels = {}

    def load_cross_validation_dataset(self, dataset_root_dir, test_group_index: int) -> Tuple[
        BlocksDataSet, BlocksDataSet]:
        root_dir = Path(dataset_root_dir)
        dirs = get_dir_directories(root_dir / 'train')

        train_dirs = [d for i, d in enumerate(dirs) if i != test_group_index]
        test_dir = dirs[test_group_index]

        train_dataset = BlocksDataSet.concatenate([self.load_dataset(d, format_=Format.Default) for d in train_dirs])
        test_dataset = self.load_dataset(test_dir, format_=Format.Default)

        self.log.write(f'*\n'
                       f'train {[d.parts[-1] for d in train_dirs]} {train_dataset}\n'
                       f'*\n'
                       f'test {test_dir.parts[-1]} {test_dataset}\n'
                       f'*')
        return train_dataset, test_dataset

    def load_dataset(self, dataset_root_dir, format_: Format) -> Union[Tuple[BlocksDataSet], BlocksDataSet]:
        root_dir = Path(dataset_root_dir)
        self.labels = {}

        self.log.write_verbose("=== Loading dataset from %s ===" % root_dir)
        if format_ == Format.Default:
            datasets = self._gather_datasets(root_dir, level=0)
            res = BlocksDataSet.concatenate(datasets)
        else:
            self.log.write_verbose('train')
            level = 0 if format_ == Format.Split else 1
            train_datasets = self._gather_datasets(root_dir / 'train', level=level)

            self.log.write_verbose('test')
            test_dataset = self._gather_datasets(root_dir / 'test', level=0)
            res = BlocksDataSet.concatenate(train_datasets), BlocksDataSet.concatenate(test_dataset)
        self.log.write_verbose("=== Dataset loading completed :D ===")

        return res

    def _gather_datasets(self, path: Path, level: int, label: int = 0):
        dirs = get_dir_directories(path)
        if not dirs:
            dss = [BlocksDataSet.from_blocks_file(file, label) for file in get_dir_csvs(path)]
            num_blocks = sum(map(len, dss))
            self.log.write_verbose("path: %s, num blocks: %d, label: %d" % (path, num_blocks, label))

            return dss

        labels = None
        if self.is_label_level(level):
            labels = sorted(list(map(lambda dir_: dir_.parts[-1], dirs)))

        datasets = []
        for d in dirs:
            if self.is_label_level(level):
                dir_name = d.parts[-1]
                label = labels.index(dir_name)

            datasets += self._gather_datasets(d, level - 1, label)

        return datasets

    @staticmethod
    def is_label_level(level: int):
        return level == 0

    def _add_directory_label(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def _get_directory_label(self, d):
        return self.labels[d]
