import sys
from collections import Counter

sys.path.append("../")
sys.path.append("./")

import torch
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
from flowpic_dataset.loader import FlowPicDataLoader
from model.FlowPicModel import FlowPicModel
from expiraments.experiment import Experiment
from training.flowpic_trainer import FlowPicTrainer


class BalancedExperiment(Experiment):
    # TODO: Implement the training loop
    def __run__(self, bs_train=128, bs_test=None, batches=100, epochs=100, early_stopping=3, checkpoints=None,
                load_checkpoint=False, lr=1e-3, reg=0, filters_per_layer=None,
                layers_per_block=2, out_classes=5, pool_every=2, drop_every=2, hidden_dims=None, ycn=False,
                label_level=1, filter_fun=0, train_portion=0.8, num_samples_per_class=0,
                **kw):
        if hidden_dims is None:
            hidden_dims = [64]
        if filters_per_layer is None:
            filters_per_layer = [10, 20]
        torch.manual_seed(self.torch_seed)

        if not bs_test:
            bs_test = max([bs_train // 4, 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_loader = FlowPicDataLoader(self.data_dir)
        ds = dataset_loader.load_dataset(label_level, filter_fun)
        ds_train, ds_test = ds.split_set(train_percent=train_portion)

        if num_samples_per_class == 0:
            label_count = Counter(ds_train.labels).values()
            num_samples_per_class = sum(label_count) // len(label_count)
        ds_train.balance(num_samples_per_class=num_samples_per_class)

        print('ds_train', ds_train)
        print('ds_test', ds_test)

        dl_train = DataLoader(ds_train, bs_train, shuffle=True)
        dl_test = DataLoader(ds_test, bs_test, shuffle=False,)

        filters = []
        for filter in filters_per_layer:
            temp = [filter] * layers_per_block
            filters += temp

        x0, _ = ds_train[0]
        model = FlowPicModel(x0.shape, out_classes, filters, hidden_dims, drop_every)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

        trainer = FlowPicTrainer(model, loss_fn, optimizer, device)

        fit_res = trainer.fit(dl_train, dl_test, epochs, checkpoints, early_stopping, print_every=5,
                              num_batches=batches, **kw)

        return fit_res


if __name__ == '__main__':
    parsed_args = BalancedExperiment.parse_cli()
    print(f'*** Starting {BalancedExperiment.__name__} with config:\n{parsed_args}')
    exp = BalancedExperiment(**vars(parsed_args))
