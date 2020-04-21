import multiprocessing
from time import time

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split

from flowpic_dataset.loader import FlowPicDataLoader
from flowpic_dataset.utils import create_dataset_weights
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    py = [1, 0, 2, 3, 2, 2, 2, 1]
    ty = [4, 2, 2, 3, 1, 0, 2, 2]
    m = confusion_matrix(ty, py)
    print(m)
    print(m.diagonal()/m.sum(1))

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
