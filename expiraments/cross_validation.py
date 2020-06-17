import sys

sys.path.append("../")
sys.path.append("./")
import argparse
import multiprocessing
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from expiraments.experiment import Experiment
from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from misc.utils import create_dir, _create_pre_trained_model, is_file, Timer
from model.flow_pic_model import FlowPicModel
from training.flowpic_trainer import FlowPicTrainer
from misc.data_classes import BatchResult
from training.trainer import Trainer


class CrossValidation(Experiment):

    def __init__(self):
        super().__init__()

    def run(self, data_dir=None, out_dir=None, bs_train=128, bs_test=256, epochs=40, early_stopping=3,
            save_checkpoint=False, load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0, filters_per_layer=None,
            layers_per_block=2, pool_every=2, drop_every=2, hidden_dims=None,
            parallel=True, k: int = None, **kw):

        out_dir = Path(out_dir)
        create_dir(out_dir)
        model_checkpoint = str(out_dir / 'model')
        cv_checkpoint = str(out_dir / 'cv')

        start_i, f1, acc, loss = 0, 0, 0, 0
        if load_checkpoint and is_file(cv_checkpoint):
            start_i, k, f1, acc, loss = self.load_cv(cv_checkpoint)

        loader = FlowCSVDataLoader()
        for i in range(start_i, k):
            ds_train, ds_test = loader.load_cross_validation_dataset(data_dir, test_group_index=i)
            dl_train = DataLoader(ds_train, bs_train, shuffle=True)
            dl_test = DataLoader(ds_test, bs_test, shuffle=True)

            input_shape = ds_train[0][0].shape
            num_classes = ds_train.get_num_classes()
            filters = self.get_filters(filters_per_layer, layers_per_block)

            model = FlowPicModel(input_shape, num_classes, filters, hidden_dims, drop_every)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

            trainer = FlowPicTrainer(model, loss_fn, optimizer, self.device, parallel)
            res = trainer.fit(dl_train, dl_test, epochs, model_checkpoint,
                              checkpoint_every=checkpoint_every,
                              save_checkpoint=save_checkpoint,
                              load_checkpoint=load_checkpoint,
                              early_stopping=early_stopping,
                              print_every=5)

            if res.num_epochs > 0:
                f1 += res.test_f1[-1]
                acc += res.test_acc[-1]
                loss += res.test_acc[-1]

            if save_checkpoint:
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

    @staticmethod
    def load_cv(cv_checkpoint: str):
        if not cv_checkpoint.endswith('.pt'):
            checkpoint_filename = f'{cv_checkpoint}.pt'
        else:
            checkpoint_filename = cv_checkpoint

        print(f'*** Loading checkpoint file {checkpoint_filename}')
        saved_state = torch.load(checkpoint_filename)
        return saved_state['i'], saved_state['k'], saved_state['f1'], saved_state['acc'], saved_state['loss']


# python expiraments/cross_validation.py --data-dir data_cv_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 40 --lr 0.001 --save-checkpoint 0 --load-checkpoint 0 --checkpoint-every 100 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 0 --verbose 0 --k 5

if __name__ == '__main__':
    t = Timer()
    t.start()
    exp = CrossValidation()
    parsed_args = exp.parse_cli()
    print(f'*** Starting {CrossValidation.__name__} with config:\n{parsed_args}')
    for i in range(10):
        exp.run(**vars(parsed_args))
        t.lap(f'{i} entire runs')
