from os import listdir
from os.path import isdir, join

from torch.utils.data.dataset import ConcatDataset

from flowpic_dataset.dataset import FlowsDataSet
from flowpic_dataset.flowpic_builder import FlowPicBuilder


class FlowPicDataLoader:
    def __init__(self, dataset_root_dir):
        self.root_dir = dataset_root_dir
        self.labels = {}

    def load_dataset(self, label_level=1, filter_fun=None):
        """
        creates a FlowDataset from each file at the leaf level of the directory tree
        and returns a ConcatDataset of all of them
        :param dataset_root_dir: path of the dataset directory
        :param filter_fun: a function that receives a directory name and returns a boolean
        :param label_level: what level of the directory tree is used as labeling
        :return: a ConcatDataset object
        """

        self.labels = {}

        if label_level == 0:
            raise ValueError("label_level must be greater then 0")

        print("=== Loading dataset from %s ===" % self.root_dir)
        dirs = [d for d in listdir(self.root_dir) if isdir(join(self.root_dir, d))]
        datasets = []
        for d in dirs:
            if filter_fun and not filter_fun(d):
                continue  # skip directory

            print("Loading %s" % d)
            label = None
            if self.__is_label__(label_level):
                label = self.__get_label__(d)

            datasets += self.__gather_datasets__(join(self.root_dir, d), filter_fun, label_level - 1, label)

        print("\n=== Dataset loading completed :D ===\n")
        return ConcatDataset(datasets)

    def __gather_datasets__(self, path, filter, label_level, label):

        dirs = [d for d in listdir(path) if isdir(join(path, d))]
        if not dirs:
            dss = [FlowsDataSet(join(path, file), label, transform=FlowPicBuilder().build_pic) for file in
                   listdir(path)]
            print(path, sum(map(len, dss)))
            return dss

        datasets = []
        for d in dirs:
            if filter and not filter(d):
                continue  # skip directory

            if label is None and self.__is_label__(label_level):
                label = self.__get_label__(d)

            datasets += self.__gather_datasets__(join(path, d), filter, label_level - 1, label)

        return datasets

    @staticmethod
    def __is_label__(label_level: int):
        return label_level == 1

    def __get_label__(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)
        return self.labels[d]
