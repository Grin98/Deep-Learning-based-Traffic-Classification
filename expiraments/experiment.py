import abc
import argparse
import json
import os
import sys
import random
from training.result_types import FitResult


class Experiment(abc.ABC):
    """
        A class abstracting the various tasks of an experiment.
        Provides methods to run, save and load an experiment

        Use parse_cli to parse the flags needed to conduct the experiment
        """

    def __init__(self, run_name, data_dir: str, out_dir='./results', seed=None, **kw):
        self.experiment_name = run_name
        self.data_dir = data_dir
        if not seed:
            seed = random.randint(0, 2 ** 31)
        self.torch_seed = seed
        self.output_dir = out_dir
        self.config = locals()
        self.result = self.__run__(**kw)

    #        self.save_experiment(self.experiment_name, self.output_dir, self.config, self.result)

    @staticmethod
    def parse_cli():
        p = argparse.ArgumentParser(description='Experiments')

        # Experiment config
        p.add_argument('--run-name', '-n', type=str,
                       help='Name of run and output file', required=True)
        p.add_argument('--data-dir', '-d', type=str, help='data folder', required=True)
        p.add_argument('--out-dir', '-o', type=str, help='Output folder',
                       default='./results', required=False)
        p.add_argument('--seed', '-s', type=int, help='Random seed',
                       default=None, required=False)

        # # Training
        p.add_argument('--bs-train', type=int, help='Train batch size',
                       default=128, metavar='BATCH_SIZE')
        p.add_argument('--bs-test', type=int, help='Test batch size',
                       metavar='BATCH_SIZE')
        p.add_argument('--batches', type=int,
                       help='Number of batches per epoch', default=10)
        p.add_argument('--epochs', type=int,
                       help='Maximal number of epochs', default=40)
        p.add_argument('--early-stopping', type=int,
                       help='Stop after this many epochs without '
                            'improvement', default=3)
        p.add_argument('--checkpoints', type=str,
                       help='Save model checkpoints to this file when test '
                            'accuracy improves', default=None)
        p.add_argument('--load-checkpoint', type=bool, default=False,
                       help='whether to start training using '
                            'the file provided in --checkpoints as starting point')
        p.add_argument('--lr', type=float,
                       help='Learning rate', default=1e-3)
        p.add_argument('--reg', type=float,
                       help='L2 regularization', default=0)

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
        # Dataset
        p.add_argument('--label-level', '-LL', type=int, default=1, metavar='LL')
        p.add_argument('--filter-fun', '-FF', type=int, default=0, metavar='FF')
        p.add_argument('--train-portion', type=float, default=0.8)
        p.add_argument('--num-samples-per-class', '-nspc', type=int, default=0)

        parsed = p.parse_args()

        return parsed

    def __run__(self,
                # Training params
                bs_train=128, bs_test=None, batches=100, epochs=100,
                early_stopping=3, checkpoints=None, load_checkpoint=False, lr=1e-3, reg=0,
                # Model params
                filters_per_layer=None, layers_per_block=2, out_classes=5, pool_every=2,
                drop_every=2, hidden_dims=None, ycn=False,
                # Dataset Loader params
                label_level=1, filter_fun=0,
                **kw):
        """
            Execute a single run of experiment with given configuration
        """
        raise NotImplementedError()

    @staticmethod
    def save_experiment(run_name, out_dir, config, fit_res):
        output = dict(
            config=config,
            results=fit_res._asdict()
        )
        output_filename = f'{os.path.join(out_dir, run_name)}.json'
        os.makedirs(out_dir, exist_ok=True)
        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f'*** Output file {output_filename} written')

    @staticmethod
    def load_experiment(filename):
        with open(filename, 'r') as f:
            output = json.load(f)

        config = output['config']
        fit_res = FitResult(**output['results'])

        return config, fit_res
