import os
from math import floor
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler


def build_pic(stream: Sequence[Tuple[float, int]],
              stream_duration_in_seconds: int = 60,
              pic_width: int = 1500,
              pic_height: int = 1500,
              ):
    # scaling stream_duration to pic's width
    x_axis_to_second_ratio = pic_width * 1.0 / stream_duration_in_seconds

    hist = torch.zeros(pic_width, pic_height)
    for packet in stream:
        # packet is (time, size)
        x_position = int(floor(float(packet[0]) * x_axis_to_second_ratio))
        y_position = packet[1]
        if x_position >= pic_width or y_position >= pic_height:
            raise Exception(f'packet position exceeded pic size of %dx%d, packet position is (%d, %d)' %
                            (pic_width, pic_height, x_position, y_position))
        hist[x_position][y_position] += 1
    return hist


def show_flow_pic(pic):
    x = pic.numpy()
    x = x.transpose()
    # changes values to 1 so that the image would be in black and white with no gray scale
    x[x >= 1] = 1
    plt.imshow(x, cmap='binary')
    plt.gca().invert_yaxis()
    plt.show()


def create_array_of_objects(x):
    a = np.empty(len(x), dtype=object)
    a[:] = x
    return a


def create_output_dir(out_root_dir: Path, input_dir_path: Path):
    sub_path = Path(*input_dir_path.parts[1:])
    out_path_dir = out_root_dir / sub_path
    create_dir(out_path_dir)
    return out_path_dir


def get_dir_directories(dir_path: Path):
    return [d for d in get_dir_items(dir_path) if d.is_dir()]


def get_dir_items(dir_path: Path):
    return list(dir_path.glob('*'))


def get_dir_csvs(dir_path: Path):
    return list(dir_path.glob('*.csv'))


def create_dir(dir_: Path):
    if not dir_.exists():
        dir_.mkdir(parents=True, exist_ok=True)


def fix_seed(seed: int):
    if seed is None:
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint: str, model_type, device):
    if not checkpoint.endswith('.pt'):
        checkpoint_filename = f'{checkpoint}.pt'
    else:
        checkpoint_filename = checkpoint
    model, best_acc, epochs_without_improvement = None, None, 0
    Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)

    if not os.path.isfile(checkpoint_filename):
        raise FileNotFoundError

    else:
        print(f'*** Loading checkpoint file {checkpoint_filename}')
        saved_state = torch.load(checkpoint_filename, map_location=device)
        best_acc = saved_state.get('best_acc', best_acc)
        epochs_without_improvement = saved_state.get('ewi', epochs_without_improvement)
        model = _create_pre_trained_model(model_type,
                                          saved_state['model_state'],
                                          saved_state['model_init_params'],
                                          device)

    return model, best_acc, epochs_without_improvement


def save_model(checkpoints, model, epoch, best_acc, epochs_without_improvement):
    checkpoint_filename = f'{checkpoints}.pt'
    saved_state = dict(best_acc=best_acc,
                       ewi=epochs_without_improvement,
                       model_state=model.state_dict(),
                       model_init_params=model.model_init_params)

    torch.save(saved_state, checkpoint_filename)
    print(f'*** Saved checkpoint {checkpoint_filename} '
          f'at epoch {epoch}')


def _create_pre_trained_model(model_class, model_state: dict, model_init_params: dict, device):
    m: nn.Module = model_class(**model_init_params)
    if device == 'cuda':
        m.to(device)
    m.load_state_dict(model_state)
    return m
