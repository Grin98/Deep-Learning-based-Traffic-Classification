import argparse
import sys
from time import time

sys.path.append("../")
sys.path.append("./")

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from flowpic_dataset.loader import FlowPicDataLoader
from model.flow_pic_model import FlowPicModel
from expiraments.experiment import Experiment
from training.flowpic_trainer import FlowPicTrainer


class SplitExperiment(Experiment):

    def run(self, data_dir=None, bs_train=128, bs_test=None, batches=100, epochs=100, early_stopping=3,
            checkpoints=None,
            load_checkpoint=False, checkpoint_every=40, lr=1e-3, reg=0, filters_per_layer=None,
            layers_per_block=2, out_classes=5, pool_every=2, drop_every=2, hidden_dims=None,
            **kw):

        if hidden_dims is None:
            hidden_dims = [64]
        if filters_per_layer is None:
            filters_per_layer = [10, 20]
        torch.manual_seed(self.torch_seed)

        if not bs_test:
            bs_test = max([bs_train // 4, 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_loader = FlowPicDataLoader(data_dir)
        ds_train, ds_test = dataset_loader.load_dataset(is_split=True)

        print('creating sampler for train')
        sampler_train = ds_train.create_weighted_random_sampler(num_to_sample=bs_train * batches, replacement=True)
        print('creating sampler for test')
        sampler_test = ds_test.create_weighted_random_sampler(num_to_sample=bs_test * batches, replacement=True)

        dl_train = DataLoader(ds_train, bs_train, shuffle=False, sampler=sampler_train)
        dl_test = DataLoader(ds_test, bs_test, shuffle=False, sampler=sampler_test)

        filters = []
        for filter in filters_per_layer:
            temp = [filter] * layers_per_block
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
                              print_every=5,
                              num_batches=batches, **kw)

        return fit_res


if __name__ == '__main__':
    exp = SplitExperiment()
    parsed_args = exp.parse_cli()
    print(parsed_args)
    print(f'*** Starting {SplitExperiment.__name__} with config:\n{parsed_args}')
    res = exp.run(**vars(parsed_args))
