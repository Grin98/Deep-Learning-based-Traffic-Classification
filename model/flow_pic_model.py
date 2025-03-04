import torch.nn as nn

from misc.output import Logger


class FlowPicModel(nn.Module):

    def __init__(self, in_size, out_classes, filters, hidden_dims, drop_every, log: Logger = Logger()):
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
        self.drop_every = drop_every
        self.log = log

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

        self.model_init_params = dict(in_size=in_size,
                                      out_classes=out_classes,
                                      filters=filters,
                                      hidden_dims=hidden_dims,
                                      drop_every=drop_every)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        # [(Conv -> ReLU) -> MaxPool]*N
        N = len(self.filters)
        for i in range(N):
            layers.append(nn.Conv2d(in_channels, self.filters[i], 10, stride=5, padding=3))
            if (i + 1) % self.drop_every == 0:
                layers.append(nn.Dropout(0.25))
            in_h = (in_h - 10 + 6) // 5 + 1
            in_w = (in_w - 10 + 6) // 5 + 1
            self.log.write_verbose("conv %dx%d" % (in_w, in_h))
            layers.append(nn.ReLU())
            in_channels = self.filters[i]
            layers.append(nn.MaxPool2d(2, stride=2))
            in_h = (in_h - 2) // 2 + 1
            in_w = (in_w - 2) // 2 + 1
            self.log.write_verbose("maxpool %dx%d" % (in_w, in_h))

        self.features = in_channels * in_h * in_w
        self.log.write_verbose("in_channels: ", in_channels)
        self.log.write_verbose("in_h: ", in_h)
        self.log.write_verbose("features: ", self.features)
        seq = nn.Sequential(*layers)
        self.log.write_verbose(seq)
        return seq

    def _make_classifier(self):
        layers = []
        # (Linear -> ReLU)*M -> softmax
        M = len(self.hidden_dims)
        input = self.features
        for i in range(M):
            layers.append(nn.Linear(input, self.hidden_dims[i]))
            layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU())
            input = self.hidden_dims[i]
        layers.append(nn.Linear(input, self.out_classes))
        layers.append(nn.Softmax(dim=1))

        seq = nn.Sequential(*layers)
        self.log.write_verbose(seq)
        self.log.write_verbose("out classes:", self.out_classes)
        return seq

    def forward(self, x):

        features = self.feature_extractor(x)
        out = self.classifier(features.view(features.shape[0], -1))

        return out

