import argparse
import sys

from torch import nn

sys.path.append("../")
sys.path.append("./")
from pathlib import Path
from time import time
from utils import create_dir
import torch
import torch.optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from flowpic_dataset.loader import FlowCSVDataLoader, PreFetchDataLoader
from model.flow_pic_model import FlowPicModel
from expiraments.experiment import Experiment
from training.flowpic_trainer import FlowPicTrainer


class SplitExperiment(Experiment):

    def run(self, data_dir=None, out_dir=None,
            bs_train=128, bs_test=None, epochs=100, early_stopping=3,
            checkpoints=None,
            load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0, filters_per_layer=None,
            layers_per_block=2, out_classes=5, pool_every=2, drop_every=2, hidden_dims=None,
            **kw):

        if out_dir is not None:
            out_dir = Path(out_dir)
            create_dir(out_dir)
            if checkpoints is not None:
                checkpoints = str(out_dir/checkpoints)

        if hidden_dims is None:
            hidden_dims = [64]

        if filters_per_layer is None:
            filters_per_layer = [10, 20]

        if not bs_test:
            bs_test = max([bs_train * 2, 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("DEVICE: ", torch.cuda.device_count())
        dataset_loader = FlowCSVDataLoader(data_dir)
        ds_train, ds_test = dataset_loader.load_dataset(is_split=True)

        dl_train = DataLoader(ds_train, bs_train, shuffle=True)
        dl_test = DataLoader(ds_test, bs_test, shuffle=True)

        filters = []
        for filter_ in filters_per_layer:
            temp = [filter_] * layers_per_block
            filters += temp

        x0, _ = ds_train[0]
        model = FlowPicModel(x0.shape, out_classes, filters, hidden_dims, drop_every)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

        trainer = FlowPicTrainer(model, loss_fn, optimizer, device)
        fit_res = trainer.fit(dl_train, dl_test, epochs, checkpoints,
                              checkpoint_every=checkpoint_every,
                              load_checkpoint=load_checkpoint,
                              early_stopping=early_stopping,
                              print_every=5, **kw)

        if out_dir is not None:
            self.save_graph(out_dir / 'loss.png', fit_res.train_loss, fit_res.test_loss, data='loss')
            self.save_graph(out_dir / 'acc.png', fit_res.train_acc, fit_res.test_acc, data='acc')
            self.save_graph(out_dir / 'f1.png', fit_res.train_f1, fit_res.test_f1, data='f1')

        return fit_res


if __name__ == '__main__':
    exp = SplitExperiment()
    parsed_args = exp.parse_cli()
    print(f'*** Starting {SplitExperiment.__name__} with config:\n{parsed_args}')
    res = exp.run(**vars(parsed_args))
