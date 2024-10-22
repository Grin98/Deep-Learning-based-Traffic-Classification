import sys

sys.path.append("../")
sys.path.append("./")
import argparse
import multiprocessing
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.experiment import Experiment
from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from misc.utils import create_dir, _create_pre_trained_model, is_file, Timer
from model.flow_pic_model import FlowPicModel
from trainers.flowpic_trainer import FlowPicTrainer
from misc.output import Logger
from misc.data_classes import BatchResult
from trainers.trainer import Trainer


class CrossValidation(Experiment):

    def __init__(self, log: Logger = Logger()):
        super().__init__()
        self.log = log

    def run(self, data_dir=None, out_dir=None, bs_train=128, bs_test=256, epochs=40, print_every=5, early_stopping=3,
            checkpoint_every=40, lr=1e-3, reg=0, filters_per_layer=None,
            layers_per_block=2, pool_every=2, drop_every=2, hidden_dims=None,
            parallel=True, k: int = None, **kw):

        out_dir = Path(out_dir)
        create_dir(out_dir)
        model_checkpoint = str(out_dir / 'model')
        cv_checkpoint = str(out_dir / 'cv')

        start_i, f1, acc, loss = 0, 0, 0, 0
        loader = FlowCSVDataLoader(self.log)
        for i in range(start_i, k):
            ds_train, ds_test = loader.load_cross_validation_dataset(data_dir, validation_group_index=i)
            dl_train = DataLoader(ds_train, bs_train, shuffle=True)
            dl_test = DataLoader(ds_test, bs_test, shuffle=True)

            input_shape = ds_train[0][0].shape
            num_classes = ds_train.get_num_classes()
            filters = self.get_filters(filters_per_layer, layers_per_block)

            model = FlowPicModel(input_shape, num_classes, filters, hidden_dims, drop_every, self.log)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

            trainer = FlowPicTrainer(model, loss_fn, optimizer, self.log, self.device, parallel)
            res = trainer.fit(dl_train, dl_test, epochs, model_checkpoint,
                              checkpoint_every, early_stopping, print_every)

            if res.num_epochs > 0:
                f1 += res.test_f1[-1]
                acc += res.test_acc[-1]
                loss += res.test_acc[-1]

            self.save_cv(cv_checkpoint, i, k, f1, acc, loss)
            self.save_fit_graphs(out_dir, res, tag=str(i))

        return f1 / k, acc / k, loss / k

    def add_parser_args(self, p: argparse.ArgumentParser):
        p = super().add_parser_args(p)
        p.add_argument('--k', type=int, help='number of folds', required=True)
        return p

    @staticmethod
    def save_cv(cv_checkpoint: str, i: int, k: int, f1: float, acc: float, loss: float):
        checkpoint_filename = f'{cv_checkpoint}.pt'
        saved_state = dict(i=i + 1, k=k, f1=f1, acc=acc, loss=loss)

        torch.save(saved_state, checkpoint_filename)
        print(f'*** Saved checkpoint to {cv_checkpoint} at fold {i + 1}/{k}')


# python experiments/cross_validation.py --data-dir data_cv_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 40 --lr 0.001 --save-checkpoint 0 --load-checkpoint 0 --checkpoint-every 100 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 0 --verbose 0 --k 5

if __name__ == '__main__':
    exp = CrossValidation()
    parsed_args = exp.parse_cli()
    print(f'*** Starting {CrossValidation.__name__} with config:\n{parsed_args}')
    exp.run(**vars(parsed_args))
