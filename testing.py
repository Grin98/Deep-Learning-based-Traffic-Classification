import csv
import multiprocessing
from collections import Counter

import torch
import numpy as np
from time import time

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split

from clasification import Classifier
from flowpic_dataset.loader import FlowPicDataLoader
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

    # p = SplitPreProcessor('.')
    # flows = p._process_file(pathlib.Path('netflix_1_half.csv'))
    # blocks = p._split_multiple_flows_to_blocks(flows)
    # with pathlib.Path('netflix_1_half_blocks.csv').open('w+', newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     p._write_blocks(blocks, writer, ' ')
    # exit()
    # p = StatisticsProcessor('classes_statistics')
    # p.process_dataset('classes_csvs')

    # x = [random.gauss(3, 1) for _ in range(10000)]
    # y = [random.gauss(4, 2) for _ in range(5000)] + [200]
    #
    # bins1 = np.linspace(-10, 10, 100)
    # bins2 = np.linspace(-20, 20, 100)
    #
    # plt.hist(x, bins1, alpha=0.5, label='x')
    # plt.hist(y, bins2, alpha=0.5, label='y')
    # plt.legend(loc='upper right')
    # plt.show()

    # d = list({0: 1, 1: 2}.items())
    # print(dict(map(lambda a: (a[0], a[1]+3), d)))
    # exit()
    # x = [[0], [1], [2], [3],
    #      [4], [5], [6], [7],
    #      [8], [9]]
    # print(x)
    # y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    # rus = RandomUnderSampler({0: 1, 1: 1, 2: 1}, replacement=True)
    # ros = RandomOverSampler({0: 5, 1: 5, 2: 2})
    # print(rus.fit_resample(x, y))
    # print(rus.sample_indices_)
    # print(ros.fit_resample(x, y))
    # exit()

    # print(random.sample(range(300), 20))
    # exit()

    # p = NoOverlapProcessor('data_tor')
    # p.process_dataset('classes_tor')
    # print('\n==========\n')
    #
    # p = NoOverlapProcessor('data_vpn')
    # p.process_dataset('classes_vpn')
    # print('\n==========\n')
    #
    # p = NoOverlapProcessor('data_reg')
    # p.process_dataset('classes_reg')
    # print('\n==========\n')

    # SplitPreProcessor('data_reg_overlap_split').process_dataset('classes_reg')
    # SplitPreProcessor('data_tor_overlap_split').process_dataset('classes_tor')
    # SplitPreProcessor('data_vpn_overlap_split').process_dataset('classes_vpn')
    # FlowPicDataLoader('./data_overlap_train', testing=True).load_dataset()
    # FlowPicDataLoader('./data_overlap_test', testing=True).load_dataset()
    # FlowPicDataLoader('./classes_csvs', testing=True).load_dataset()
    # FlowPicDataLoader('./data_reg_overlap_split', testing=True).load_dataset()
    FlowPicDataLoader('./data_tor_overlap_split', testing=True).load_dataset(is_split=True)
    # p = FlowPicDataLoader('./data_vpn_overlap_split', testing=False)
    # tr, te = p.load_dataset(is_split=True)

    # l = FlowPicDataLoader('./data_tor', testing=False)
    # ds = l.load_dataset()
    # print(ds)
    # ds.balance(num_samples_per_class=350)
    # print(ds)
    # print('\n==========\n')

    # l = FlowPicDataLoader('./data_overlap', testing=True)
    # l.load_dataset()
    # print('\n==========\n')
    # exit()
    # l = FlowPicDataLoader('./data_overlap_test', testing=True)
    # l.load_dataset()
    # print('\n==========\n')

    ''' ========================================================================================== '''

    # p = OverlapProcessor('data_overlap_test', 'data_overlap_train')
    # p.process_dataset('classes_csvs')
    # print('\n==========\n')
    #
    # l = FlowPicDataLoader('./data_overlap_test', testing=True)
    # l.load_dataset()
    # print('\n==========\n')
    # #
    # l = FlowPicDataLoader('./data_overlap_train', testing=True)
    # l.load_dataset()
    # print('\n==========\n')
    #
    # l = FlowPicDataLoader('./data_overlap', testing=True)
    # l.load_dataset()
    # print('\n==========\n')
    # #
    # l = FlowPicDataLoader('./classes_csvs', testing=True)
    # l.load_dataset()
    # print('\n==========\n')

    # flows = np.array([(1, 'a'), (2, 'b')])
    # num_flows = len(flows)
    # test_indices = random.sample(range(num_flows), 1)
    # train = flows
    # test = flows[test_indices]
    # train = np.delete(train, test_indices)
    # print(test_indices)
    # print(test)
    # print(train)

# d = FlowsDataSet('./data_overlap/voip/reg/iscx_voip_vpn.raw.csv')
# x, y = d[0]
# print(y)

# l = FlowPicDataLoader('./data')
# l.load_dataset()

# ty = [4, 2, 2, 3, 1, 0, 2, 2]
# py = [1, 0, 2, 3, 2, 2, 2, 1]
#
# m = confusion_matrix(ty, py)
# s = f1_score(ty, py, average='weighted')
# print(s)
# print(m)
# print(m.diagonal()/m.sum(1))
#
# l = FlowPicDataLoader('./classes_csvs')
# l.load_dataset()

# s = WeightedRandomSampler([0.2]*10, 6, replacement=True)
# i1 = iter(s)
# i2 = iter(s)
# i3 = iter(s)
# while True:
#     print(next(i1), next(i2), next(i3))

# f = lambda d: d != 'tor'
# l = FlowPicDataLoader('./data')
# ds = l.load_dataset()
#
# dataset_length = len(ds)
# label_probabilities = l.get_label_weights()
# print(label_probabilities)
# train_length = int(dataset_length * 0.8)
# test_length = dataset_length - train_length
#
# ds_train, ds_test = random_split(ds, (train_length, test_length))
#
# print("creating weights")
# start = time()
# w = create_dataset_weights(ds_train, l.get_label_weights())
# print(time() - start)
# print(len(ds_train), len(w))
# print(w)

# x = {'a': 2, 'b': 5}
# v = list(x.values())
# print(type(v), v)
