import abc
import os
import sys
from operator import add
from time import time

import tqdm
import torch

from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Any

from misc.data_classes import EpochResult, FitResult, BatchResult
from misc.loging import Logger
from misc.utils import save_model, load_model, is_file


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model: nn.Module, loss_fn, optimizer, log: Logger, device: torch.device = None, parallel: bool = True):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model: nn.Module = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.log = log
        self.device = device

        if self.device:
            if torch.cuda.device_count() > 1 and parallel:
                self.log.write_verbose('using parallel on model')
                model = nn.DataParallel(model)
            model.to(self.device)

    def fit(self, dl_train, dl_test: DataLoader,
            num_epochs, model_checkpoint: str,
            checkpoint_every: int,
            save_checkpoint: bool,
            load_checkpoint: bool,
            early_stopping: int,
            print_every: int) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param save_checkpoint: Whether to save a checkpoint or not
        :param load_checkpoint: Whether to load a saved checkpoint or not
        :param checkpoint_every: Number of epochs for every checkpoint save
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param model_checkpoint: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        train_loss, train_acc, train_f1, test_loss, test_acc, test_f1 = [], [], [], [], [], []

        best_acc = None
        epochs_without_improvement = 0
        start_epoch = 1
        if is_file(model_checkpoint) and load_checkpoint:
            self.model, best_acc, start_epoch, epochs_without_improvement = load_model(model_checkpoint, type(self.model), self.device)

        for epoch in range(start_epoch, num_epochs + 1):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs or epoch == 1:
                verbose = True
                self.log.write(f'--- EPOCH {epoch}/{num_epochs} ---')

            loss, acc, f1 = self.train_epoch(dl_train, verbose)
            train_loss.append(sum(loss).item() / float(len(loss)))
            train_acc.append(acc.item())
            train_f1.append(f1)

            loss, acc, f1 = self.test_epoch(dl_test, verbose)
            test_loss.append(sum(loss).item() / float(len(loss)))
            test_acc.append(acc.item())
            test_f1.append(f1)

            epochs_without_improvement = epochs_without_improvement + 1

            if model_checkpoint is not None and save_checkpoint and epoch % checkpoint_every == 0:
                save_model(model_checkpoint, self.model, epoch, best_acc, epochs_without_improvement)

            if best_acc is None or acc > best_acc:
                best_acc = acc
                epochs_without_improvement = 0

            if early_stopping is not None and epochs_without_improvement == early_stopping:
                break

        actual_num_epochs = (num_epochs + 1) - start_epoch
        return FitResult(actual_num_epochs,
                         train_loss, train_acc, train_f1,
                         test_loss, test_acc, test_f1)

    def train_epoch(self, data_loader_train, verbose: bool) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param data_loader_train: DataLoader for the training set.
        :param verbose: whether to print info of epoch or not
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(data_loader_train, self.train_batch, verbose)

    def test_epoch(self, data_loader_test, verbose: bool) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param data_loader_test: DataLoader for the test set.
        :param verbose: whether to print info of epoch or not
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(data_loader_test, self.test_batch, verbose)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    def _foreach_batch(self,
                       data_loader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_total = 0
        f1_scores = []
        f1_per_class = None
        num_batches = len(data_loader)

        if verbose:
            pbar_file = self.log.file
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(data_loader)
            for batch_idx in range(num_batches):
                # s = time()
                batch = next(dl_iter)
                # f = time()
                # print('batch fetch time', f - s)
                # s = time()
                batch_res = forward_fn(batch)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_total += batch_res.num_total
                num_correct += batch_res.num_correct
                f1_scores.append(batch_res.f1_score)
                if f1_per_class is None:
                    f1_per_class = batch_res.f1_per_class
                else:
                    f1_per_class = list(map(add, f1_per_class, batch_res.f1_per_class))
                # f = time()
                # print('batch process time', f - s)

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_total
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_f1_per_class = [round(f / num_batches, ndigits=2) for f in f1_per_class]
            pbar.set_description(f'{pbar_name} '
                                 f'(Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f}, '
                                 f'F1 {avg_f1:.3f}), ')
                                 # f'F1 Classes {avg_f1_per_class}')

        return EpochResult(losses=losses, accuracy=accuracy, f1=avg_f1)
