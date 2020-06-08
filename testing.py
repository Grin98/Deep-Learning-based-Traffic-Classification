from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import torch
from pyshark.packet.packet import Packet
from torch.utils.data import DataLoader

from flowpic_dataset.dataset import FlowDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from flowpic_dataset.processors import SplitPreProcessor, BasicProcessor
from misc.data_classes import Flow
from misc.utils import show_flow_pic, is_file
from pcap_extraction.pcap_flow_extractor import PcapParser

# python expiraments/split_experiment.py --data-dir data_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 35 --lr 0.001 --save-checkpoint 1 --load-checkpoint 0 --checkpoint-every 1 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1

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

    ds = FlowCSVDataLoader().load_dataset('data_reg/train')
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    it = iter(dl)
    s = time()
    b, _ = next(it)
    f = time()
    print(f - s)
    print(len(b))

    exit()
    f = Path('classes_reg/video/reg/netflix_1.csv')
    # f = Path('classes_reg/video/reg/netflix_2.csv')
    # f3 = Path('parsed_flows/netflix_3.csv')
    # f4 = Path('parsed_flows/netflix_4.csv')
    #
    # ds1 = FlowDataSet.from_flows_file(f1)
    # ds2 = FlowDataSet.from_flows_file(f2)
    # # ds3 = FlowDataSet.from_flows_file(f3)
    # # ds4 = FlowDataSet.from_flows_file(f4)
    # print(len(ds1))
    # print(len(ds2))


    # p = BasicProcessor(60, 15, 1500)
    # n1 = p.process_file_to_flows(f)[0]
    # # n2 = p.process_file_to_flows(f2)[0]
    #
    # mask = n1.times < 600
    # a = n1.times[mask]
    # s = n1.sizes[mask]
    #
    # n1 = Flow(n1.app, n1.five_tuple, n1.start_time, num_packets=len(a), times=a, sizes=s)
    # print(n1)
    # print(n1.times[-1], n1.sizes[-1])
    # PcapParser.write_flow_rows(f, [n1])
    # n1 = p.process_file_to_flows(f)[0]
    # print("=================")
    # print(n1)
    # print(n1.times[-1], n1.sizes[-1])
    # ds = FlowDataSet.from_flows_file(f)
    # print(len(ds))
    # exit()

    # a, b = FlowCSVDataLoader('data_reg_net').load_dataset(is_split=True)
    # print(a.data)
    # x = [1, 432, 1040, 439, 902, 10]
    # for i in x:
    #     print(a[i])
    #     show_flow_pic(a[i][0].squeeze(0))
    #     show_flow_pic(b[i][0].squeeze(0))
    #     print('pic =', i)
    #     print(a.data[i])
    #     print(a.labels[i])
    #     print(b.data[i])
    #     print(b.labels[i])
    #
    # print('a start', a.start_times)
    # print('a labels', a.labels)
    # print('a data', a.data)
    # print('b start', b.start_times)
    # print('b labels', b.labels)
    # print('b data', b.data)
    # exit()

    # exit()
    # p = SplitPreProcessor('data_reg_net', test_percent=0.3, train_size_cap=2100, test_size_cap=900).process_dataset('classes_reg')
    # FlowCSVDataLoader('data_reg_net', verbose=True).load_dataset(is_split=True)
    # FlowCSVDataLoader('data_reg', verbose=True).load_dataset(is_split=True)
    # ds = FlowsDataSet.from_blocks_file(p)
    # print(len(ds))
    # exit()
