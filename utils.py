import os
from pathlib import Path
from typing import List

import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
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


def load_model(checkpoint: str, model_type, device):
    if not checkpoint.endswith('.pt'):
        checkpoint_filename = f'{checkpoint}.pt'
    else:
        checkpoint_filename = checkpoint
    model, best_acc, epochs_without_improvement = None, None, 0
    Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)

    if os.path.isfile(checkpoint_filename):
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
    if device:
        m.to(device)
    m.load_state_dict(model_state)
    return m