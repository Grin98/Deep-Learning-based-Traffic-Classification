import csv
import multiprocessing
from collections import Counter

import torch
import numpy as np
from time import time

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split

from clasification import Classifier
from flowpic_dataset.loader import FlowCSVDataLoader
from flowpic_dataset.processors import SplitPreProcessor, NoOverlapPreProcessor, StatisticsProcessor
from flowpic_dataset.dataset import FlowsDataSet
from sklearn.metrics import confusion_matrix, f1_score
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pathlib
import os
import matplotlib.pyplot as plt


class C:
    def __init__(self, x):
        self.x = x

    def p(self):
        print(self.x)


if __name__ == '__main__':

    y = [1, 2, 3, 2, 1]
    pred = [1, 2, 3, 1, 1]

    pr = set(y+pred)
    ls = range(5)
    fs = list(f1_score(y, pred, average=None))
    print([fs[i] if i in pr else None for i in ls])
