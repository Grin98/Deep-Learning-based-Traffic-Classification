from collections import Counter

import torch
from training.trainer import Trainer
from training.result_types import BatchResult
from sklearn.metrics import confusion_matrix, f1_score


class FlowPicTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)
        out = self.model(X)
        self.optimizer.zero_grad()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        values, _ = torch.max(out, dim=1)
        num_correct = values - out[range(out.shape[0]), y]
        num_correct = torch.where(num_correct == 0, torch.ones_like(num_correct), torch.zeros_like(num_correct))
        num_correct = sum(num_correct)

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            out = self.model(X)
            loss = self.loss_fn(out, y)
            values, pred = torch.max(out, dim=1)
            num_correct = values - out[range(out.shape[0]), y]
            num_correct = torch.where(num_correct == 0, torch.ones_like(num_correct), torch.zeros_like(num_correct))
            num_correct = sum(num_correct)

            y = y.cpu()
            pred = pred.cpu()
            # m = confusion_matrix(y, pred)
            def_s = f1_score(y, pred, average=None)
            weighted_s = f1_score(y, pred, average='weighted')
            macro_s = f1_score(y, pred, average='macro')
            print('f1 per class:', def_s)
            print('weighted average:', weighted_s)
            print('average:', macro_s)

        return BatchResult(loss, num_correct)
