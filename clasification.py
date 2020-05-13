import torch


class Classifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def classify(self, X):
        if self.device:
            X = X.to(self.device)

        out = self.model(X)
        values, pred = torch.max(out, dim=1)

        print('out', out)
        print('vals', values)
        print('pred', pred)
