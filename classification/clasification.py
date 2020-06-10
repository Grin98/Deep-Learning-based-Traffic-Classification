from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from misc.data_classes import ClassifiedBlock
from model.flow_pic_model import FlowPicModel
from misc.utils import load_model, fix_seed


class Classifier:
    def __init__(self, model, device, seed: int = 42):
        self.model = model
        self.device = device
        self.model.train(False)
        fix_seed(seed)

    def classify(self, X):
        if self.device == 'cuda':
            X = X.to(self.device)

        out = self.model(X)
        values, pred = torch.max(out, dim=1)

        # print('out', out)
        # print('vals', values)
        # print('pred', pred)
        return pred

    def classify_folder(self, path: str, label: int, tag=''):
        ds = FlowCSVDataLoader(path, verbose=False).load_dataset()
        self.classify_dataset(ds, label, tag)

    def classify_dataset(self, ds: BlocksDataSet, batch_size: int = 256, label: int = 0, tag: str = ''):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        cnt = Counter([])
        dl_iter = iter(dl)
        classified_blocks = []
        for j in range(len(dl)):
            x, _ = next(dl_iter)
            pred = self.classify(x)
            pred = pred.cpu().tolist()
            cnt += Counter(pred)
            classified_blocks += [ClassifiedBlock(ds.get_block(j * batch_size + i), pred[i]) for i in range(len(pred))]

        return cnt, classified_blocks

    def classify_folders(self, p1:Path, folders: Sequence[str], p2: Path):
        for i, folder in enumerate(folders):
            file_samples = p1 / folder / p2
            self.classify_folder(str(file_samples), label=i, tag=folder)




if __name__ == '__main__':
    device = 'cuda'
    folders = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
    p1 = Path('data_reg_overlap_split/test')
    p2 = Path('reg')
    file_checkpoint = '../reg_overlap_split'
    f = Path('../parsed_flows/netflix_4.csv')

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = Classifier(model, device)
    ds = BlocksDataSet.from_flows_file(f, 1)
    a, _ = c.classify_dataset(ds, 1, tag='fb-chat')
    print(a)

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


