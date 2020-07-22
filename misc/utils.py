import csv
import itertools
import os
from math import floor
from multiprocessing import Lock
from pathlib import Path
from string import Template
from time import time
from typing import List, Sequence, Tuple, overload, Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler

from misc.constants import BLOCK_DURATION, PACKET_SIZE_LIMIT
from misc.data_classes import Flow, ClassifiedBlock
from misc.output import Logger


class Timer:
    """
    class for a more convenient way to check run time of code
    """
    def __init__(self):
        self._s = 0

    def start(self):
        self._s = time()

    def lap(self, tag: str = ''):
        f = time()
        diff = format(f - self._s, '.2f')
        print(f'{tag} time = {diff}[sec]')


def build_pic(data: Sequence[Tuple[float, int]]):
    """
    builds FlowPic from a data and returns it
    note: see README for definitions
    """
    stream_duration = BLOCK_DURATION
    pic_width = PACKET_SIZE_LIMIT
    pic_height = PACKET_SIZE_LIMIT

    # scaling stream_duration to pic's width
    packets = np.array(data)
    x_axis_to_second_ratio = pic_width * 1.0 / stream_duration
    packets[:, 0] *= x_axis_to_second_ratio
    packets = np.floor(packets)
    max_x = np.max(packets[:, 0])
    max_y = np.max(packets[:, 1])
    if max_x > pic_width or max_y > pic_height:
        raise Exception(f'Packets are out of range of histogram max_x={max_x}, max_y={max_y}')
    hist, _, _ = np.histogram2d(x=packets[:, 0], y=packets[:, 1],
                                bins=[range(pic_width + 1), range(pic_height + 1)])

    return torch.from_numpy(hist).float()


def show_flow_pic(pic):
    """
    displays a FlowPic
    """
    x = pic.numpy()
    x = x.transpose()
    # changes values to 1 so that the image would be in black and white with no gray scale
    x[x >= 1] = 1
    plt.imshow(x, cmap='binary')
    plt.gca().invert_yaxis()
    plt.show()


def create_array_of_objects(x):
    """
    :param x: array-like object
    :return: numpy array of x
    """
    a = np.empty(len(x), dtype=object)
    a[:] = x
    return a


def create_output_dir(out_root_dir: Path, input_dir_path: Path):
    """
    creates and returns a new directory where its path is identical to input_dir_path except for the root directory
    which is replaced with out_root_dit.
    """
    sub_path = Path(*input_dir_path.parts[1:])
    out_path_dir = out_root_dir / sub_path
    create_dir(out_path_dir)
    return out_path_dir


def create_multiple_output_dirs(out_root_dirs: Sequence[Path], input_dir_path: Path):
    return [create_output_dir(d, input_dir_path) for d in out_root_dirs]


def get_dir_directories(dir_: Path):
    """
    returns the path to all the dirs in dir_
    """
    return [d for d in get_dir_items(dir_) if d.is_dir()]


def get_dir_items(dir_: Path):
    """
    returns all the items (dirs and files) in dir_
    """
    return list(dir_.glob('*'))


def get_dir_csvs(dir_: Path):
    """
    returns all the csv files in dir_
    """

    return list(dir_.glob('*.csv'))


def get_dir_pcaps(dir_: Path):
    """
    returns all the pcap files in dir_
    """

    return list(dir_.glob('*.pcap')) + list(dir_.glob('*.pcapng'))


def create_dir(dir_: Path):
    if not dir_.exists():
        dir_.mkdir(parents=True, exist_ok=True)


def is_file(file_path: str):
    return os.path.isfile(file_path)


def set_seed(seed: int):
    """
    sets seed for the pseudo random generators of numpy and torch
    and forces torch to be deterministic
    """

    if seed is None:
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint: str, model_type, device: str):
    """
    loads the saved model from checkpoint
    :param checkpoint: file of a saved model
    :param model_type: the class of the model
    :param device: either cpu or cuda
    :return: the saved model and some additional data
    """

    if not checkpoint.endswith('.pt'):
        checkpoint_filename = f'{checkpoint}.pt'
    else:
        checkpoint_filename = checkpoint

    print(f'*** Loading checkpoint file {checkpoint_filename}')
    model, best_acc, last_epoch, epochs_without_improvement = None, None, 0, 0
    saved_state = torch.load(checkpoint_filename, map_location=device)
    best_acc = saved_state.get('best_acc', best_acc)
    last_epoch = saved_state.get('last_epoch', last_epoch)
    epochs_without_improvement = saved_state.get('ewi', epochs_without_improvement)
    model = _create_pre_trained_model(model_type,
                                      saved_state['model_state'],
                                      saved_state['model_init_params'],
                                      device)

    return model, best_acc, last_epoch, epochs_without_improvement


def save_model(checkpoints: str, model, epoch: int, best_acc: float, epochs_without_improvement: float):
    """
    saves the model and additional data to a .pt file
    important: model must have model_init_params attribute that is the init parameters with which it was created
    """
    checkpoint_filename = f'{checkpoints}.pt'
    saved_state = dict(best_acc=best_acc,
                       last_epoch=epoch + 1,
                       ewi=epochs_without_improvement,
                       model_state=model.state_dict(),
                       model_init_params=model.model_init_params)

    torch.save(saved_state, checkpoint_filename)
    print(f'*** Saved checkpoint {checkpoint_filename} '
          f'at epoch {epoch}')


def _create_pre_trained_model(model_class, model_state: dict, model_init_params: dict, device):
    """
    :param model_class: the class of the model to be created
    :param model_state: model_state as defined by torch
    :param model_init_params: a dict that is passed to the constructor of the model
    :param device: either cpu or cuda
    :return: object of type model_class
    """

    m: nn.Module = model_class(**model_init_params)
    if device == 'cuda':
        m.to(device)
    m.load_state_dict(model_state)
    return m


def write_flows(writable, flows: Sequence[Flow]):
    """
    writes flows to a file
    :param writable: either a path to a file or an object with writerows method
    :param flows: the flows to write to a file
    """

    rows = [f.convert_to_row() for f in flows]
    if isinstance(writable, Path):
        with writable.open(mode='w+', newline='') as out:
            writer = csv.writer(out, delimiter=',')
            writer.writerows(rows)
    else:
        writable.writerows(rows)


def chain_list_of_tuples(l):
    return list(itertools.chain.from_iterable(l))


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def move_item(ls: Union[List, Tuple], from_index: int, to_index: int):
    item = ls[from_index]
    item = [item] if isinstance(ls, list) else (item,)
    if to_index < from_index:
        ls = ls[:to_index] + item + ls[to_index: from_index] + ls[from_index + 1:]
    else:
        ls = ls[:from_index] + ls[from_index + 1:to_index + 1] + item + ls[to_index + 1:]
    return ls
