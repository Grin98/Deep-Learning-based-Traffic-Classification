from pathlib import Path
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
