import torch.nn as nn


class FlowPicModel(nn.Module):

    def __init__(self, in_size, out_classes, filters, hidden_dims):
        """
        :param in_size: Size of input pixel images, (Packet size, arrival time).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, _ = tuple(self.in_size)

        layers = []
        # [(Conv -> ReLU) -> MaxPool]*N
        N = len(self.filters)
        for i in range(N):
            layers.append(nn.Conv2d(in_channels, self.filters[i], kernel_size=10, stride=5))
            layers.append(nn.ReLU())
            in_channels = self.filters[i]
            layers.append(nn.MaxPool2d(kernel_size=150))
        self.features = in_channels * in_channels

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # (Linear -> ReLU)*M -> softmax
        M = len(self.hidden_dims)
        input = self.features
        for i in range(M):
            layers.append(nn.Linear(input, self.hidden_dims[i]))
            layers.append(nn.ReLU())
            input = self.hidden_dims[i]
        layers.append(nn.Linear(input, self.out_classes))
        layers.append(nn.Softmax())

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):

        features = self.feature_extractor(x)
        out = self.classifier(features.view(features.shape[0], -1))

        return out
