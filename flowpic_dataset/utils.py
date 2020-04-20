from typing import List

import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler


def show_flow_pic(pic):
    x = pic.numpy()
    x = x.transpose()
    # changes values to 1 so that the image would be in black and white with no gray scale
    x[x >= 1] = 1
    plt.imshow(x, cmap='binary')
    plt.gca().invert_yaxis()
    plt.show()


def create_dataset_weights(dataset: Dataset, label_probabilities: List[float]):
    return [label_probabilities[y] for _, y in dataset]


def create_under_sampling_sampler(dataset: Dataset, bach_size: int, label_probabilities: List[float]):
    return WeightedRandomSampler(create_dataset_weights(dataset, label_probabilities),
                                 bach_size*10,
                                 replacement=True)
