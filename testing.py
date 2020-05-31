from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pyshark.packet.packet import Packet

from flowpic_dataset.dataset import FlowDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from flowpic_dataset.processors import SplitPreProcessor
from misc.utils import show_flow_pic
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

    a = FlowDataSet.from_blocks_file('data_reg/train/chat/reg/data.csv')
    print(a.data[2])
    print(a[2])
    show_flow_pic(a[311][0].squeeze(0))
    t: torch.tensor = a[2][0]
    for i, row in enumerate(t[0]):
        for j, x in enumerate(row):
            if x.item() > 0:
                print(i, j, ' ', x.item())
    exit()

    # SplitPreProcessor('delete').process_dataset('classes_reg')
    a, b = FlowCSVDataLoader('data_reg').load_dataset(is_split=True)
    print(a.data)
    x = [1, 432, 1040, 439, 902, 10]
    for i in x:
        print(a[i])
        show_flow_pic(a[i][0].squeeze(0))
        show_flow_pic(b[i][0].squeeze(0))
        print('pic =', i)
        print(a.data[i])
        print(a.labels[i])
        print(b.data[i])
        print(b.labels[i])

    print('a start', a.start_times)
    print('a labels', a.labels)
    print('a data', a.data)
    print('b start', b.start_times)
    print('b labels', b.labels)
    print('b data', b.data)
    exit()

    exit()
    # p = 'data_reg_overlap_split/train/video/reg/data.csv'
    # FlowCSVDataLoader('data_reg_overlap_split', verbose=True).load_dataset(is_split=True)
    # # ds = FlowsDataSet.from_blocks_file(p)
    # # print(len(ds))
    # exit()
