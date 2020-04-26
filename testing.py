import multiprocessing
from collections import Counter

import numpy as np
from time import time

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split

from flowpic_dataset.loader import FlowPicDataLoader
from flowpic_dataset.utils import create_dataset_weights
from flowpic_dataset.preprocessor import PreProcessor
from flowpic_dataset.dataset import FlowsDataSet
from sklearn.metrics import confusion_matrix, f1_score


class C:
    def __init__(self, x):
        self.x = x

    def p(self):
        print(self.x)


if __name__ == '__main__':
    # a = {'a': 2, 'b': 3}
    # b = {'b': 1}
    # c = {'a': 4, 'c': 1}
    # input = [a, b, c]
    # s = sum(
    #     (Counter(x) for x in input),
    #     Counter())
    # print(s)

    # TPS = 60  # TimePerSession in secs
    # DELTA_T = 15  # Delta T between splitted sessions
    # MIN_TPS = 40
    # MIN_LENGTH = 10
    #
    # ts = np.array([0, 0.34, 1.2, 3.4, 10.2, 10.3, 15.34, 20, 22.3, 59, 65, 75, 76, 120, 200], dtype=float)
    # sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int)
    #
    # print(int(ts[-1] / DELTA_T - TPS / DELTA_T) + 1)
    # for t in range(int(ts[-1] / DELTA_T - TPS / DELTA_T) + 1):
    #     mask = ((ts >= t * DELTA_T) & (ts <= (t * DELTA_T + TPS)))
    #     ts_mask = ts[mask]
    #     sizes_mask = sizes[mask]
    #
    #     print(ts_mask, sizes_mask)
    #     print(mask)


    p = PreProcessor('./classes_csvs', './overlapped_data')
    p.process_dataset()

    # print('\n==========\n')
    #
    # l = FlowPicDataLoader('./overlapped_data', testing=True)
    # l.load_dataset()
    # print('\n==========\n')
    #
    # d = FlowsDataSet('./overlapped_data/voip/reg/iscx_voip_vpn.raw.csv', FlowsDataSet.Label.Category)
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
