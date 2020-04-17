import sys

sys.path.append("../")
sys.path.append("./")

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from flowpic_dataset.dataset import load_dataset
from model.FlowPicModel import FlowPicModel
from training.experiment import Experiment
from training.flowpic_trainer import FlowPicTrainer


class FlowPicExperiment(Experiment):
    # TODO: Implement the training loop
    def __run__(self, bs_train=128, bs_test=None, batches=100, epochs=100, early_stopping=3, checkpoints=None,
                load_checkpoint=False, lr=1e-3, reg=1e-3, filters_per_layer=None,
                layers_per_block=2, out_classes=5, pool_every=2, drop_every=2, hidden_dims=None, ycn=False,
                **kw):
        if hidden_dims is None:
            hidden_dims = [64]
        if filters_per_layer is None:
            filters_per_layer = [10, 20]
        torch.manual_seed(self.torch_seed)

        if not bs_test:
            bs_test = max([bs_train // 4, 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = load_dataset(self.data_dir)
        dataset_length = len(dataset)
        train_length = int(dataset_length * 0.8)
        ds_train, ds_test = random_split(dataset, (train_length, dataset_length-train_length))
        dl_train = DataLoader(ds_train, bs_train, shuffle=True)
        dl_test = DataLoader(ds_test, bs_test, shuffle=True)

        filters = []
        for filter in filters_per_layer:
            temp = [filter] * layers_per_block
            filters += temp

        x0, _ = ds_train[0]
        model = FlowPicModel(x0.shape, out_classes, filters, hidden_dims, drop_every)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = FlowPicTrainer(model, loss_fn, optimizer, device)

        fit_res = trainer.fit(dl_train, dl_test, epochs, checkpoints, early_stopping, print_every=5,
                              max_batches=batches, **kw)

        return fit_res


if __name__ == '__main__':
    parsed_args = FlowPicExperiment.parse_cli()
    print(f'*** Starting {FlowPicExperiment.__name__} with config:\n{parsed_args}')
    exp = FlowPicExperiment(**vars(parsed_args))
