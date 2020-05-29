import abc
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from typing import List

from misc.utils import fix_seed


class Experiment(abc.ABC):
    """
        A class abstracting the various tasks of an experiment.
        Provides methods to run, save and load an experiment

        Use parse_cli to parse the flags needed to conduct the experiment
        """

    def __init__(self, seed=42):

        self.torch_seed = seed
        fix_seed(seed)

    #        self.save_experiment(self.experiment_name, self.output_dir, self.config, self.result)

    def add_parser_args(self, p: argparse.ArgumentParser):
        # Experiment config
        p.add_argument('--data-dir', '-d', type=str, help='data folder', required=True)
        p.add_argument('--out-dir', '-o', type=str, help='Output folder',
                       default=None, required=False)
        # p.add_argument('--seed', '-s', type=int, help='Random seed',
        #                default=None, required=False)
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
        p.add_argument('--checkpoints', type=str,
                       help='Save model checkpoints to this file when test '
                            'accuracy improves', default=None)
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
        p.add_argument('--out-classes', '-O', type=int,
                       help='Number of output classes', default=5)
        p.add_argument('--pool-every', '-P', type=int, metavar='P',
                       help='Pool after this number of conv layers')
        p.add_argument('--drop-every', '-D', type=int, metavar='P',
                       help='Pool after this number of conv layers', default=2)
        p.add_argument('--hidden-dims', '-H', type=int, nargs='+',
                       help='Output size of hidden linear layers',
                       metavar='H')

        return p

    def parse_cli(self):
        p = argparse.ArgumentParser(description=type(self).__name__)
        p = self.add_parser_args(p)
        parsed = p.parse_args()
        return parsed

    @abc.abstractmethod
    def run(self,
                # Training params
                data_dir=None, out_dir=None,
                bs_train=128, bs_test=None, epochs=100,
                early_stopping=3, checkpoints=None, load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0,
                # Model params
                filters_per_layer=None, layers_per_block=2, out_classes=5, pool_every=2,
                drop_every=2, hidden_dims=None, **kw):
        """
            Execute a single run of experiment with given configuration
        """
        raise NotImplementedError()

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
