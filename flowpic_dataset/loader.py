from pathlib import Path
from queue import Queue
from threading import Thread
from time import time
from typing import Tuple, Union

from torch.utils.data import DataLoader

from flowpic_dataset.dataset import FlowsDataSet

from utils import get_dir_directories, get_dir_csvs


class FlowCSVDataLoader:
    def __init__(self, dataset_root_dir, testing=False):
        self.root_dir = Path(dataset_root_dir)
        self.testing = testing
        self.labels = {}

    def load_dataset(self, is_split: bool = False) -> Union[Tuple[FlowsDataSet], FlowsDataSet]:
        """
        creates a FlowDataset from each file at the leaf level of the directory tree
        and returns a ConcatDataset of all of them
        """

        self.labels = {}
        print("=== Loading dataset from %s ===" % self.root_dir)
        if is_split:
            print('train')
            train_datasets = self._gather_datasets(self.root_dir/'train')

            print('test')
            test_dataset = self._gather_datasets(self.root_dir/'test')
            res = FlowsDataSet.concatenate(train_datasets), FlowsDataSet.concatenate(test_dataset)
        else:
            datasets = self._gather_datasets(self.root_dir)
            res = FlowsDataSet.concatenate(datasets)
        print("=== Dataset loading completed :D ===\n")

        return res

    def _gather_datasets(self, path: Path, label_level: bool = True, label: int = None):
        dirs = get_dir_directories(path)
        if not dirs:
            dss = [FlowsDataSet(file, label, testing=self.testing) for file in get_dir_csvs(path)]
            num_blocks = sum(map(len, dss))
            print("path: %s, num blocks: %d, label: %d" % (path, num_blocks, label))

            return dss

        labels = None
        if label_level:
            labels = sorted(list(map(lambda dir_: dir_.parts[-1], dirs)))

        datasets = []
        for d in dirs:
            if label_level:
                dir_name = d.parts[-1]
                label = labels.index(dir_name)

            datasets += self._gather_datasets(d, False, label)

        return datasets

    def _add_directory_label(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def _get_directory_label(self, d):
        return self.labels[d]


class PreFetchDataLoader:
    """
    Wrapper for objects of type DataLoader.
    starts a separate thread for fetching data when iterated
    """
    def __init__(self, dl: DataLoader):
        self.dl = dl
        self.q = Queue(maxsize=1)
        self.i = 0

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        self.i = 0
        fetcher = PreFetchDataLoader.Fetcher(self.dl, self.q)
        fetcher.start()
        return self

    def __next__(self):
        if self.i >= len(self.dl):
            raise StopIteration

        self.i += 1
        return self.q.get(block=True)

    class Fetcher(Thread):

        def __init__(self, dl: DataLoader, q: Queue):
            super().__init__()
            self.dl = dl
            self.q = q

        def run(self):
            iter_ = iter(self.dl)
            for i in range(len(self.dl)):
                self.q.put(next(iter_), block=True)






