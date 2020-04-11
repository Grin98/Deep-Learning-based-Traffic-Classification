from os import listdir
from os.path import join, isdir

from torch.utils.data.dataset import Dataset, ConcatDataset
import csv
from typing import List
import numpy as np
import torch
from flowpic_dataset.flowpic_builder import FlowPicBuilder


def load_dataset(dataset_root_dir):
    """
    creates a FlowDataset from each file at the leaf level of the directory tree
    and returns a ConcatDataset of all of them
    """

    print("Loading dataset")
    dirs = [d for d in listdir(dataset_root_dir) if isdir(join(dataset_root_dir, d))]
    datasets = []
    for d in dirs:
        print("Loading %s" % d)
        datasets += __gather_datasets__(join(dataset_root_dir, d), d)

    print("\nDataset loading completed\n")
    return ConcatDataset(datasets)


def __gather_datasets__(path, label):
    """
    iterates on the directory hierarchy and gathers the datasets
    """

    dirs = [d for d in listdir(path) if isdir(join(path, d))]
    if not dirs:
        return [FlowsDataSet(join(path, file), label, transform=FlowPicBuilder().build_pic) for file in listdir(path)]

    datasets = []
    for d in dirs:
        datasets += __gather_datasets__(join(path, d), label)
    return datasets


'''
parameter label will be the label of all the data entries in the file
'''


class FlowsDataSet(Dataset):
    def __init__(self, csv_file_path, label, transform=None):
        self.label = label
        self.transform = transform

        with open(csv_file_path, newline='') as f:
            self.data = [self.__transform_row_to_flow__(row) for row in csv.reader(f, delimiter=',')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not torch.is_tensor(idx):
            idx = [idx]

        res = []
        for i in idx:
            x = self.data[i]
            if self.transform:
                x = self.transform(x)
            res.append((x, self.label))

        if len(res) == 1:
            return res[0]
        return res

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
