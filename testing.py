import csv
import itertools
from collections import Counter
from math import floor
from pathlib import Path
from time import time, sleep

import numpy as np
import torch
from pyshark.packet.packet import Packet
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from classification.clasifiers import Classifier, FlowCsvClassifier
from experiments.cross_validation import CrossValidation
from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader, Format
from flowpic_dataset.processors import HoldOutPreProcessor, BasicProcessor, CrossValidationPreProcessor, get_dir_csvs
from gui.graph_frame import FlowPicGraphFrame
from misc.constants import BLOCK_DURATION, BLOCK_INTERVAL
from misc.data_classes import Flow
from misc.output import Logger, Progress
from misc.utils import show_flow_pic, is_file, Timer, load_model, get_dir_items
from model.flow_pic_model import FlowPicModel
from pcap_extraction.aggregation import Aggregator
from pcap_extraction.pcap_flow_extractor import PcapParser


# python experiments/cross_validation.py --data-dir data_cv_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 40 --lr 0.001 --save-checkpoint 0 --load-checkpoint 0 --checkpoint-every 100 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 0 --verbose 1 --k 5

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

    ls = [0, 1, 2, 3, 4, 5, 6, 7]
    from_index: int = 0
    to_index: int = 0

    item = ls[from_index]
    if to_index < from_index:
        ls = ls[:to_index] + [item] + ls[to_index: from_index] + ls[from_index + 1:]
    else:
        ls = ls[:from_index] + ls[from_index + 1:to_index+1] + [item] + ls[to_index+1:]

    print(ls)
