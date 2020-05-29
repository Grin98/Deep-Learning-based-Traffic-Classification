from collections import Counter
from pathlib import Path

import numpy as np
from pyshark.packet.packet import Packet

from flowpic_dataset.loader import FlowCSVDataLoader
from flowpic_dataset.processors import SplitPreProcessor
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

    # a = [(1, 2, (3, 4)), (5, 6, (7, 8))]
    # print(list(zip(*a)))
    # exit()

    # print(dir(Packet))

    # a = ([(1, 0.2), (1, 0.2), (1, 0.2), (1, 0.2)],)
    # b = np.array(a)
    # print(a)
    # print(b)

    # file = Path('pcaps')/'netflix_1.pcapng'
    # print(file.is_file())
    #
    # p = PcapParser(4)
    # f = p.parse_file(file)
    # a = list(map(lambda x: x[7], f))
    # print(a)

    # a = {1: [100, 2, 300, 1, 0],
    #      2: [8, 5, 6, 1],
    #      3: [300, 5000]}
    # b = nlargest(4, a, key=lambda x: len(a.get(x)))
    # print(a, b)
    #
    a, b = FlowCSVDataLoader('delete').load_dataset(is_split=True)
    print(a[0])
    print('start', a.start_times)
    print('labels', a.labels)
    print('data', a.data)
    exit()

    SplitPreProcessor('delete').process_dataset('classes_reg')
    exit()
    # p = 'data_reg_overlap_split/train/video/reg/data.csv'
    # FlowCSVDataLoader('data_reg_overlap_split', verbose=True).load_dataset(is_split=True)
    # # ds = FlowsDataSet.from_blocks_file(p)
    # # print(len(ds))
    # exit()
