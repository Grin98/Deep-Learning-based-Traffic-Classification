from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from sklearn.metrics import f1_score
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

        # print('out', out)
        # print('vals', values)
        # print('pred', pred)
        return pred

    def classify_folders(self, p1:Path, folders: Sequence[str], p2: Path):

        for i, folder in enumerate(folders):
            file_samples = p1 / folder / p2
            ds = FlowsDataSet(file_samples, global_label=i)
            dl = DataLoader(ds, batch_size=128, shuffle=False)

            cnt = Counter([])
            dl_iter = iter(dl)
            for j in range(len(dl)):
                x, y = next(dl_iter)
                pred = c.classify(x)
                pred = pred.cpu()
                pred = pred.tolist()
                cnt += Counter(pred)

            total = len(ds)
            print(folder, ' acc = ', round(cnt[i] / total, 2))
            print(cnt, 'total =', total)



if __name__ == '__main__':
    device = 'cuda'
    folders = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
    p1 = Path('data_reg_overlap_split/train')
    p2 = Path('reg/data.csv')
    file_checkpoint = 'reg_overlap_split'

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = Classifier(model, device)
    c.classify_folders(p1, folders, p2)

    # ds = FlowsDataSet(file_samples, global_label=3)
    # dl = DataLoader(ds, batch_size=128, shuffle=True)
    #
    # cnt = Counter([])
    # f = 0
    # dl_iter = iter(dl)
    # for j in range(len(dl)):
    #     x, y = next(dl_iter)
    #     pred = c.classify(x)
    #     pred = pred.cpu()
    #     pred = pred.tolist()
    #     cnt += Counter(pred)
    # print('total =', len(ds))
    # print('f1 =', f/len(dl))
    # print(cnt)


