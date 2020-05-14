from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flowpic_dataset.dataset import FlowsDataSet
from model.flow_pic_model import FlowPicModel
from utils import load_model


class Classifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def classify(self, X):
        if self.device:
            X = X.to(self.device)

        out = self.model(X)
        values, pred = torch.max(out, dim=1)

        print('out', out)
        print('vals', values)
        print('pred', pred)


if __name__ == '__main__':
    device = 'cuda'
    file_samples = Path('netflix_1_half_blocks.csv')
    file_checkpoint = 'reg_overlap_split'

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    ds = FlowsDataSet(file_samples)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    c = Classifier(model, device)

    dl_iter = iter(dl)
    for batch_idx in range(len(ds)):
        x, _ = next(dl_iter)
        c.classify(x)


