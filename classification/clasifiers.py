import itertools
from collections import Counter
from pathlib import Path
from typing import Sequence, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset

from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.loader import FlowCSVDataLoader
from flowpic_dataset.processors import BasicProcessor
from misc.constants import UNKNOWN_THRESHOLD
from misc.data_classes import ClassifiedBlock, Flow, ClassifiedFlow
from misc.output import Progress
from model.flow_pic_model import FlowPicModel
from misc.utils import load_model, set_seed
from pcap_extraction.pcap_flow_extractor import PcapParser

LOADING_BAR_UPDATE_INTERVAL = 2


class Classifier:
    """

    """
    def __init__(self, model, device: str, seed: int = 42, progress=Progress()):
        """

        :param model: the model to use when classifying
        :param device: either cpu or cuda
        :param seed: seed for the pseudo random generators
        :param progress: used for showing progress in the gui
        """
        self.model = model
        self.device = device
        self.progress = progress
        self.model.train(False)
        set_seed(seed)

    def classify(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: batch of FlowPics to classify
        :return: the actual pred and the probabilities for the different categories of each block
        """
        if self.device == 'cuda':
            X = X.to(self.device)

        probabilities = self.model(X)
        _, pred = torch.max(probabilities, dim=1)
        return pred, probabilities

    def classify_multiple_flows(self, flows: Sequence[Flow]) -> Sequence[ClassifiedFlow]:
        """
        classifies the given flows
        """
        self.progress.counter_title('classified flows').set_counter(0)
        num_flows = len(flows)
        res = []
        for i, f in enumerate(flows):
            if i % LOADING_BAR_UPDATE_INTERVAL == 0:
                self.progress.set_counter(i, num_flows)
            res.append(self.classify_flow(f))
        return res

    def classify_flow(self, f: Flow) -> ClassifiedFlow:
        """
        classifies the flow
        """
        ds = BlocksDataSet.from_flows([f])
        distribution, classified_blocks = self.classify_dataset(ds)
        pred = self.get_pred(distribution)
        return ClassifiedFlow(f, pred, classified_blocks)

    def classify_dataset(self, ds: BlocksDataSet, batch_size: int = 128):
        """
        classifies the blocks in a dataset and returns the number of predictions of each category
        and the classified blocks.
        """

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        cnt = Counter([])
        dl_iter = iter(dl)
        classified_blocks = []
        for j in range(len(dl)):
            x, _ = next(dl_iter)
            pred, prob = self.classify(x)
            pred = pred.cpu().tolist()
            prob = prob.cpu().tolist()
            cnt += Counter(pred)
            classified_blocks += [ClassifiedBlock(ds.get_block(j * batch_size + i), pred[i], prob[i]) for i in range(len(pred))]

        return cnt, classified_blocks

    @staticmethod
    def get_pred(dist: Counter):
        """
        returns -1(aka unknown) if the UNKNOWN_THRESHOLD is passed otherwise returns the most common prediction
        """

        common = dist.most_common(n=2)
        if len(common) == 1:
            return common[0][0]

        c1, c2 = common
        ratio = c2[1] / c1[1]
        if ratio > UNKNOWN_THRESHOLD:
            return -1
        return c1[0]


class PcapClassifier:
    """
    class for classifying pcap files
    """

    def __init__(self, model,
                 device: str,
                 seed: int = 42,
                 progress=Progress()
                 ):
        """
        :param model: model that will be used for classifying FlowPics
        :param device: if value is cuda then the model will run on gpu and if cpu then it will run on cpu
        :param seed: seed for numpy and torch
        """
        self.progress = progress
        self.classifier = Classifier(model, device, seed, self.progress)
        self.parser = PcapParser(self.progress)

    def classify_file(self, file: Path, num_flows_to_classify: int) -> Sequence[ClassifiedFlow]:
        """

        :param file: the file to classify
        :param num_flows_to_classify: how many flows in the pcap to classify (takes the flows with the most packets)
        :return: the classified flows
        """
        self.progress.reset()
        self.progress.progress_title(f'parsing file {str(file.name)}')
        flows = self.parser.parse_file(file, num_flows_to_classify)
        self.progress.reset()
        self.progress.progress_title(f'classifying file {str(file.name)}')
        return self.classifier.classify_multiple_flows(flows)

    def classify_multiple_files(self, files: Sequence[Path], num_flows_to_classify: int = 1) -> List[ClassifiedFlow]:
        classified_flows = list(itertools.chain.from_iterable([self.classify_file(f, num_flows_to_classify)
                                                               for f in files]))
        return classified_flows


class FlowCsvClassifier:
    """
    class for classifying csv files
    """

    def __init__(self, model,
                 device: str,
                 seed: int = 42,
                 progress=Progress()
                 ):
        """

        :param model: model that will be used for classifying FlowPics
        :param device: if value is cuda then the model will run on gpu and if cpu then it will run on cpu
        :param seed: seed for numpy and torch
        """
        self.progress = progress
        self.classifier = Classifier(model, device, seed, self.progress)
        self.processor = BasicProcessor()

    def classify_file(self, file: Path) -> Sequence[ClassifiedFlow]:
        self.progress.reset()
        self.progress.progress_title(f'parsing file {str(file.name)}')
        flows = self.processor.process_file_to_flows(file)
        self.progress.reset()
        self.progress.progress_title(f'classifying {str(file.name)}')
        return self.classifier.classify_multiple_flows(flows)

    def classify_multiple_files(self, files: Sequence[Path]) -> List[ClassifiedFlow]:
        classified_flows = list(itertools.chain.from_iterable([self.classify_file(f)
                                                               for f in files]))
        return classified_flows


