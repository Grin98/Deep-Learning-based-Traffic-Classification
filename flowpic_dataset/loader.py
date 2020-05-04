from os import listdir
from os.path import isdir, join

from torch.utils.data.dataset import ConcatDataset

from flowpic_dataset.dataset import FlowsDataSet
from flowpic_dataset.flowpic_builder import FlowPicBuilder
import numpy as np
import torch


class FlowPicDataLoader:
    def __init__(self, dataset_root_dir, testing=False):
        self.root_dir = dataset_root_dir
        self.testing = testing
        self.labels = {}
        self.labels_count = {}

        # possible filters
        def everything(_): return True

        def reg(d): return d != 'tor' and d != 'vpn'

        def tor(d): return d != 'reg' and d != 'vpn'

        def vpn(d): return d != 'reg' and d != 'tor' and d != 'browsing'

        self.filters = [everything, reg, tor, vpn]
        self.filter_fun = self.filters[0]

    def get_label_weights(self):
        quantities = list(self.labels_count.values())
        return [1.0 / x for x in quantities]

    def get_num_labels(self):
        return len(self.labels)

    def add_possible_filter(self, filter_fun):
        """
        :param filter_fun: a function of type (str) -> bool
        :return: index of function in filters
        """
        index = len(self.filters)
        self.filters.append(filter_fun)
        return index

    def load_dataset(self, label_level=1, filter_fun=0):
        """
        creates a FlowDataset from each file at the leaf level of the directory tree
        and returns a ConcatDataset of all of them

        :param filter_fun: an index of a filtering function that receives a directory name and returns a boolean
        :param label_level: what level of the directory tree is used as labeling
        :return: a ConcatDataset object of all files in the leaf level
        """
        if not (0 <= filter_fun < len(self.filters)):
            raise ValueError("filter_fun is out of range")

        if label_level <= 0:
            raise ValueError("label_level must be greater then 0")

        self.labels = {}
        self.labels_count = {}
        self.filter_fun = self.filters[filter_fun]

        print("=== Loading dataset from %s ===" % self.root_dir)
        datasets = self.__gather_datasets__(self.root_dir, label_level - 1, None)
        print("=== Dataset loading completed :D ===\n")
        return FlowsDataSet.concatenate(datasets)

    def __gather_datasets__(self, path, label_level, label):

        dirs = [d for d in listdir(path) if isdir(join(path, d))]
        if not dirs:
            dss = [FlowsDataSet(join(path, file), label, testing=self.testing) for file in
                   listdir(path)]
            num_flows = sum(map(len, dss))
            self.__add_to_count__(num_flows, label)
            print("path: %s, num entries: %d, label: %d" % (path, num_flows, label))

            return dss

        datasets = []
        for d in dirs:
            if not self.filter_fun(d):
                continue  # skip directory

            if self.__is_label__(label_level):
                self.__add_directory_label__(d)
                label = self.__get_directory_label__(d)

            datasets += self.__gather_datasets__(join(path, d), label_level - 1, label)

        return datasets

    @staticmethod
    def __is_label__(label_level: int):
        return label_level == 0

    def __add_directory_label__(self, d: str):
        if d not in self.labels:
            self.labels[d] = len(self.labels)

    def __remove_directory_label__(self, label: str):
        if label in self.labels:
            i = self.labels.pop(label)
            if i in self.labels_count:
                self.labels_count.pop(i)

    def __get_directory_label__(self, d):
        return self.labels[d]

    def __add_to_count__(self, num: int, label: int):
        if label not in self.labels_count:
            self.labels_count[label] = num
        else:
            self.labels_count[label] += num
