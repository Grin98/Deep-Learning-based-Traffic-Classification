import argparse
import multiprocessing
import os
import sys
from time import time

sys.path.append("../")
sys.path.append("./")
from pathlib import Path
from misc.utils import create_dir
import torch
import torch.optim
from torch.utils.data import DataLoader
from flowpic_dataset.loader import FlowCSVDataLoader, Format
from model.flow_pic_model import FlowPicModel
from expiraments.experiment import Experiment
from training.flowpic_trainer import FlowPicTrainer
from misc.loging import Logger


class SplitExperiment(Experiment):

    def __init__(self, log: Logger = Logger()):
        super().__init__()
        self.log = log

    def run(self, data_dir=None, out_dir=None, data_format=0,
            bs_train=128, bs_test=None, epochs=100, early_stopping=3, print_every=5,
            save_checkpoint=False, load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0, filters_per_layer=None,
            layers_per_block=2, pool_every=2, drop_every=2, hidden_dims=None,
            parallel=True, **kw):

        model_checkpoint = None
        out_dir = Path(out_dir)
        create_dir(out_dir)
        if save_checkpoint or load_checkpoint:
            model_checkpoint = str(out_dir/'model')

        dataset_loader = FlowCSVDataLoader()
        self.timer.start()
        ds_train, ds_test = dataset_loader.load_dataset(data_dir, self.int_to_format(data_format))
        self.timer.lap('dataset loading')
        dl_train = DataLoader(ds_train, bs_train, shuffle=True)
        dl_test = DataLoader(ds_test, bs_test, shuffle=True)

        input_shape = ds_train[0][0].shape
        num_classes = ds_train.get_num_classes()
        filters = self.get_filters(filters_per_layer, layers_per_block)

        model = FlowPicModel(input_shape, num_classes, filters, hidden_dims, drop_every)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

        trainer = FlowPicTrainer(model, loss_fn, optimizer, self.log, self.device, parallel)
        self.timer.start()
        fit_res = trainer.fit(dl_train, dl_test, epochs, model_checkpoint,
                              checkpoint_every,
                              save_checkpoint,
                              load_checkpoint,
                              early_stopping,
                              print_every)
        self.timer.lap('training')

        if save_checkpoint:
            self.save_fit_graphs(out_dir, fit_res)

        return fit_res

    def add_parser_args(self, p: argparse.ArgumentParser):
        p = super().add_parser_args(p)
        p.add_argument('--data-format', type=int,
                       help='what is the dataset folder structure, possible values are:\n'
                            '0 for split to train and test folders\n'
                            '1 for split to train and test folders and train is divided further'
                            'to sub folders like in cross validation', required=True)
        return p

    @staticmethod
    def int_to_format(x):
        return {0: Format.Split,
                1: Format.SplitCV}[x]

# python expiraments/split_experiment.py --data-dir data_cv_net_reg1 --out-dir del --data-format 1 --bs-train 128 --bs-test 256 --epochs 40 --print-every 5 --lr 0.001 --reg 0.0001 --save-checkpoint 1 --load-checkpoint 0 --checkpoint-every 50 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 1 --verbose 1


if __name__ == '__main__':
    exp = SplitExperiment()
    parsed_args = exp.parse_cli()
    print(f'*** Starting {SplitExperiment.__name__} with config:\n{parsed_args}')
    res = exp.run(**vars(parsed_args))
