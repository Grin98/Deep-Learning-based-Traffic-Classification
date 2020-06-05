from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Tuple, Union

from torch.utils.data import DataLoader

from flowpic_dataset.dataset import FlowDataSet

from misc.utils import get_dir_directories, get_dir_csvs


class FlowCSVDataLoader:
    def __init__(self, testing=False, verbose: bool = True):
        self.testing = testing
        self.verbose = verbose
        self.labels = {}

    def load_cross_validation_dataset(self, dataset_root_dir, test_group_index: int) -> Tuple[FlowDataSet, FlowDataSet]:
        root_dir = Path(dataset_root_dir)
        dirs = get_dir_directories(root_dir)

        test_dataset = self.load_dataset(dir([test_group_index]), is_split=False)
        train_datasets = []
        for i, d in enumerate(dirs):
            if i != test_group_index:
                train_datasets += self.load_dataset(d, is_split=False)

        return FlowDataSet.concatenate(train_datasets), test_dataset

    def load_dataset(self, dataset_root_dir, is_split: bool = False) -> Union[Tuple[FlowDataSet], FlowDataSet]:
        root_dir = Path(dataset_root_dir)
        self.labels = {}

        self._print_verbose("=== Loading dataset from %s ===" % root_dir)
        if is_split:
            self._print_verbose('train')
            train_datasets = self._gather_datasets(root_dir/'train')

            self._print_verbose('test')
            test_dataset = self._gather_datasets(root_dir/'test')
            res = FlowDataSet.concatenate(train_datasets), FlowDataSet.concatenate(test_dataset)
        else:
            datasets = self._gather_datasets(root_dir)
            res = FlowDataSet.concatenate(datasets)
        self._print_verbose("=== Dataset loading completed :D ===\n")

        return res

    def _gather_datasets(self, path: Path, label_level: bool = True, label: int = 0):
        dirs = get_dir_directories(path)
        if not dirs:
            dss = [FlowDataSet.from_blocks_file(file, label) for file in get_dir_csvs(path)]
            num_blocks = sum(map(len, dss))
            self._print_verbose("path: %s, num blocks: %d, label: %d" % (path, num_blocks, label))

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

    def _print_verbose(self, s: str):
        if self.verbose:
            print(s)


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






