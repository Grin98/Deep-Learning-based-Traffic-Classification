from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Tuple, Union, Sequence

from torch.utils.data import DataLoader

from flowpic_dataset.dataset import BlocksDataSet
from misc.output import Logger

from misc.utils import get_dir_directories, get_dir_csvs


class Format(Enum):
    Default = 0  # the sub folders of the root folder are the labels of the flows
    Split = 1  # the root folder has train and test as sub folders
    SplitCV = 2  # like Split but train is divided further to groups(folds)


class FlowCSVDataLoader:
    """
    class for traversing directories tree and creating datasets for training
    """
    def __init__(self, log: Logger = Logger()):
        self.log = log
        self.labels = {}

    def load_dataset(self, dataset_root_dir: str, format_: Format) -> Union[Tuple[BlocksDataSet, BlocksDataSet], BlocksDataSet]:
        """
        depending on the structure(aka format) of the root_dir directories tree, it returns 1 or 2 datasets.
        if 2 then one is the train set and the second is the test set.
        """
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

    def load_cross_validation_dataset(self, dataset_root_dir, validation_group_index: int) -> Tuple[BlocksDataSet, BlocksDataSet]:
        """
        assumes that the structure of the root_dir directories tree is of Format.SplitCV, i.e. root_dir has 2 sub folders,
        train and test, where train also has sub folders which are the various groups(folds) for the cross validation.
        The method returns 2 datasets (train and validation) where the flows are gather only from the train sub folder.
        The validation dataset is consisted only from one group which is at index validation_group_index.
        """
        root_dir = Path(dataset_root_dir)
        dirs = get_dir_directories(root_dir / 'train')

        train_dirs = [d for i, d in enumerate(dirs) if i != validation_group_index]
        test_dir = dirs[validation_group_index]

        train_dataset = BlocksDataSet.concatenate([self.load_dataset(d, format_=Format.Default) for d in train_dirs])
        test_dataset = self.load_dataset(test_dir, format_=Format.Default)

        self.log.write(f'*\n'
                       f'train {[d.parts[-1] for d in train_dirs]} {train_dataset}\n'
                       f'*\n'
                       f'test {test_dir.parts[-1]} {test_dataset}\n'
                       f'*')
        return train_dataset, test_dataset

    def _gather_datasets(self, path: Path, level: int, label: int = 0) -> Sequence[BlocksDataSet]:
        """
        recursively iterates through the directories DFS style and gathers the flows from the csvs at the leaf level.
        The labels are determent by the folder at depth equal to level + 1.
        For example let's assume a csv file where the path is root -> chat -> reg -> data.csv and level is 0 then the
        label of all the flows in data.csv will be labeled as chat.

        :param path: current folder
        :param level: the level in the tree which will be used as the label for the flows
        :param label: current label of the path
        :return: a sequence of datasets
        """
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
    def is_label_level(level: int) -> bool:
        return level == 0

    def _add_directory_label(self, d: str):
        """
        adds the name of a folder to possible labels
        """
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def _get_directory_label(self, d: str) -> int:
        """
        returns the appropriate label for the name of a folder
        """
        return self.labels[d]
