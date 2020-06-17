import abc
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from typing import List

import torch
from misc import utils

from misc.data_classes import FitResult
from misc.utils import fix_seed, Timer
from model.flow_pic_model import FlowPicModel


class Experiment(abc.ABC):
    """
        A class abstracting the various tasks of an experiment.
        Provides methods to run, save and load an experiment

        Use parse_cli to parse the flags needed to conduct the experiment
        """

    def __init__(self, seed=42):

        self.timer = Timer()
        fix_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #        self.save_experiment(self.experiment_name, self.output_dir, self.config, self.result)

    @abc.abstractmethod
    def run(self,
            # Training params
            data_dir=None, out_dir=None,
            bs_train=128, bs_test=None, epochs=100,
            early_stopping=3, save_checkpoint=False, load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0,
            # Model params
            filters_per_layer=None, layers_per_block=2, pool_every=2,
            drop_every=2, hidden_dims=None,
            parallel=True, verbose=True, **kw):
        """
            Execute a single run of experiment with given configuration
        """
        raise NotImplementedError()

    def parse_cli(self):
        p = argparse.ArgumentParser(description=type(self).__name__)
        p = self.add_parser_args(p)
        parsed = p.parse_args()
        utils.verbose = parsed.verbose
        return parsed

    def add_parser_args(self, p: argparse.ArgumentParser):
        """
        adds possible arguments to be accepted in command line
        """
        # Experiment config
        p.add_argument('--data-dir', '-d', type=str, help='data folder', required=True)
        p.add_argument('--out-dir', '-o', type=str, help='Output folder', required=True)

        # # Training
        p.add_argument('--bs-train', type=int, help='Train batch size',
                       default=128, metavar='BATCH_SIZE')
        p.add_argument('--bs-test', type=int, help='Test batch size',
                       default=256, metavar='BATCH_SIZE')
        p.add_argument('--epochs', type=int,
                       help='Maximal number of epochs', default=40)
        p.add_argument('--early-stopping', type=int,
                       help='Stop after this many epochs without '
                            'improvement', default=None)
        p.add_argument('--save-checkpoint', type=int,
                       help='Save model checkpoints to this file when test '
                            'accuracy improves', default=0)
        p.add_argument('--load-checkpoint', type=int, default=0,
                       help='whether to start training using '
                            'the file provided in --checkpoints as starting point')
        p.add_argument('--checkpoint-every', type=int, default=40,
                       help='once in how many epochs to save the model')
        p.add_argument('--lr', type=float,
                       help='Learning rate', default=1e-3)
        p.add_argument('--reg', type=float,
                       help='L2 regularization', default=0)
        # checkpoint_every
        # # Model
        p.add_argument('--filters-per-layer', '-K', type=int, nargs='+',
                       help='Number of filters per conv layer in a block',
                       metavar='K')
        p.add_argument('--layers-per-block', '-L', type=int, metavar='L',
                       help='Number of layers in each block', default=1)
        p.add_argument('--pool-every', '-P', type=int, metavar='P',
                       help='Pool after this number of conv layers')
        p.add_argument('--drop-every', '-D', type=int, metavar='P',
                       help='Pool after this number of conv layers', default=2)
        p.add_argument('--hidden-dims', '-H', type=int, nargs='+',
                       help='Output size of hidden linear layers',
                       metavar='H')
        p.add_argument('--parallel', type=int, help='if 1 then parallels model if possible', default=1)
        p.add_argument('--verbose', type=int, help='if 1 then run prints additional info', default=1)

        return p

    @staticmethod
    def get_filters(filters_per_layer, layers_per_block):
        filters = []
        for filter_ in filters_per_layer:
            temp = [filter_] * layers_per_block
            filters += temp
        return filters

    def save_fit_graphs(self, out_dir: Path, fit_res: FitResult, tag: str = ''):
        self.save_graph(out_dir / f'loss{tag}.png', fit_res.train_loss, fit_res.test_loss, data='loss')
        self.save_graph(out_dir / f'acc{tag}.png', fit_res.train_acc, fit_res.test_acc, data='acc')
        self.save_graph(out_dir / f'f1{tag}.png', fit_res.train_f1, fit_res.test_f1, data='f1')

    @staticmethod
    def save_graph(file: Path, train: List[float], test: List[float], data: str = ''):
        title = 'Training and Validation' + ' ' + data
        y = data
        if file is not None:
            epochs = range(1, len(train) + 1)
            plt.plot(epochs, train, 'g', label='Training')
            plt.plot(epochs, test, 'b', label='validation')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel(y)
            plt.legend()
            plt.savefig(str(file))
            plt.clf()
