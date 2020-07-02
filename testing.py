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

from classification.clasifiers import Classifier
from expiraments.cross_validation import CrossValidation
from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader, Format
from flowpic_dataset.processors import HoldOutPreProcessor, BasicProcessor, CrossValidationPreProcessor, get_dir_csvs
from gui.graph_frame import FlowPicGraphFrame
from misc.data_classes import Flow
from misc.output import Logger, Progress
from misc.utils import show_flow_pic, is_file, Timer, load_model
from model.flow_pic_model import FlowPicModel
from pcap_extraction.pcap_flow_extractor import PcapParser


# python expiraments/cross_validation.py --data-dir data_cv_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 40 --lr 0.001 --save-checkpoint 0 --load-checkpoint 0 --checkpoint-every 100 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 0 --verbose 1 --k 5

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

    # cat = 'voip'
    # file = Path(f'tagged_flows_reg/{cat}.csv')
    # flows = []
    # with file.open(newline='') as f_in:
    #     data = csv.reader(f_in, delimiter=',')
    #     for i, row in enumerate(data):
    #         if i == 100:
    #             break
    #         f = Flow.create_from_row(row)
    #         flows.append(f)
    #
    # PcapParser.write_flow_rows(Path(f'example/{cat}.csv'), flows)

    y = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         2,
         1]
    pred = [1, 2, 1, 2, 3, 1, 1, 1, 2]
    c = Counter(y)
    print(Classifier.get_pred(c))
    # ls = np.unique(y).tolist()
    # tot = f1_score(y, pred, average='weighted', labels=ls)
    # pc = f1_score(y, pred, average=None, labels=ls)
    # print(tot)
    # print(pc)
    exit()


    # _, b = FlowPicGraphFrame.f1_score(y, pred, labels=[0, 1, 2, 3, 4])
    # print(a, b)
    # exit()
    # print(list(map(lambda d: str(d),get_dir_csvs(Path('data_netflix')))))
    # dss = [BlocksDataSet.from_flows_file(d) for d in get_dir_csvs(Path('classes_netflix'))]
    # [print(ds) for ds in dss]

    # p = CrossValidationPreProcessor('data_cv_nz_reg2', test_percent=0.2,
    #                                 train_size_cap=2400, test_size_cap=600, k=5)
    # p.process_dataset(dataset_dir='classes_reg')
    # # FlowCSVDataLoader().load_dataset('data_cv_net_reg', format_=Format.SplitCV)
    # # print('*****************')
    # exit()

    # lr = list(np.logspace(start=-3, stop=-1, num=3))
    # reg = list(np.logspace(start=-4, stop=-1, num=4))
    # reg.append(0.0)
    # conf = list(itertools.product(lr, reg))
    # print(lr, reg)
    # print(len(conf), conf)
    # c = CrossValidation()
    # c.run('data_cv_reg', 'del', early_stopping=None, save_checkpoint=True,
    #       load_checkpoint=True, filters_per_layer=[10, 20], layers_per_block=1,hidden_dims=[64], k=5)
    # l = FlowCSVDataLoader()
    # train, test = l.load_dataset('data_cv_net_reg1', Format.SplitCV)
    # print(train)
    # print(test)
    # print(train, test)
    # print('===================')
    # train, test = l.load_dataset('del', format_=Format.SplitCV)
    # print(train, test)
    exit()
    # p = CrossValidationPreProcessor('data_cv_reg',
    #                                 test_percent=0.2,
    #                                 train_size_cap=2400,
    #                                 test_size_cap=600,
    #                                 k=5)
    #
    # # p1.process_dataset('classes_reg')
    # p.process_dataset('classes_reg')

    exit()
    # stream = [(0, 0), (0, 1), (0, 2), (0, 3)]
    # packets = np.array(stream)
    # packets[:, 0] *= 1
    # packets = np.floor(packets)
    # hist, _, _ = np.histogram2d(x=packets[:, 0], y=packets[:, 1], bins=[range(1501), range(1501)])
    #
    # print(hist.shape)
    # print(hist[1499])
    # a = np.array(stream)
    # a[:, 0] *= 5
    # # b = np.floor(a[:][:])
    # print(a, a.shape)
    # print(b)
    ds = BlocksDataSet.from_blocks_file(Path('data_reg/test/chat/reg/data.csv'))
    x, _ = ds[0]
    print(x.shape)
    # dl = DataLoader(ds, batch_size=128, shuffle=True)
    # it = iter(dl)
    # s = time()
    # b, _ = next(it)
    # f = time()
    # print(f - s)
    # print(b.shape)

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
