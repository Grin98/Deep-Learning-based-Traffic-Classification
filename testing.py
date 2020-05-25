import csv
import multiprocessing
from collections import Counter
from heapq import nlargest

import torch
import numpy as np
from time import time, sleep

from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.data import random_split

from clasification import Classifier
from flowpic_dataset.loader import FlowCSVDataLoader, PreFetchDataLoader
from flowpic_dataset.processors import SplitPreProcessor, StatisticsProcessor, BasicProcessor
from flowpic_dataset.dataset import FlowsDataSet
from sklearn.metrics import confusion_matrix, f1_score
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
import os
import matplotlib.pyplot as plt

from pcap_extraction.pcap_flow_extractor import PcapParser


class C:
    def __init__(self, x):
        self.x = x
        self.ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.it = iter(self.ls)
        self.i = 0

    def p(self):
        print(self.x)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.ls):
            raise StopIteration

        self.i += 1
        return next(self.it)


if __name__ == '__main__':
    file = Path('pcaps')/'netflix_1.pcapng'
    print(file.is_file())

    p = PcapParser()
    f = p.parse_file(file, num_flows_to_return=4)
    a = list(map(lambda x: x[7], f))
    print(a)

    # a = {1: [100, 2, 300, 1, 0],
    #      2: [8, 5, 6, 1],
    #      3: [300, 5000]}
    # b = nlargest(4, a, key=lambda x: len(a.get(x)))
    # print(a, b)
    #
    # SplitPreProcessor('delete').process_dataset('classes_reg')
    # exit()
    # p = 'data_reg_overlap_split/train/video/reg/data.csv'
    # FlowCSVDataLoader('data_reg_overlap_split', verbose=True).load_dataset(is_split=True)
    # # ds = FlowsDataSet.from_blocks_file(p)
    # # print(len(ds))
    # exit()
