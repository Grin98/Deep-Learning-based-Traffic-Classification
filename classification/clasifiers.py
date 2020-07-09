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
    def __init__(self, model, device, seed: int = 42, progress=Progress()):
        self.model = model
        self.device = device
        self.progress = progress
        self.model.train(False)
        set_seed(seed)

    def classify(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.device == 'cuda':
            X = X.to(self.device)

        probabilities = self.model(X)
        _, pred = torch.max(probabilities, dim=1)
        return pred, probabilities

    def classify_multiple_flows(self, flows: Sequence[Flow]) -> Sequence[ClassifiedFlow]:
        self.progress.counter_title('classified flows').set_counter(0)
        num_flows = len(flows)
        res = []
        for i, f in enumerate(flows):
            if i % LOADING_BAR_UPDATE_INTERVAL == 0:
                self.progress.set_counter(i, num_flows)
            res.append(self.classify_flow(f))
        return res

    def classify_flow(self, f: Flow) -> ClassifiedFlow:
        ds = BlocksDataSet.from_flows([f])
        distribution, classified_blocks = self.classify_dataset(ds)
        pred = self.get_pred(distribution)
        return ClassifiedFlow(f, pred, classified_blocks)

    def classify_dataset(self, ds: BlocksDataSet, batch_size: int = 128):
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
        common = dist.most_common(n=2)
        if len(common) == 1:
            return common[0][0]

        c1, c2 = common
        ratio = c2[1] / c1[1]
        if ratio > UNKNOWN_THRESHOLD:
            return -1
        return c1[0]


class PcapClassifier:

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


if __name__ == '__main__':
    device = 'cpu'
    file_checkpoint = '../reg_overlap_split'
    f = Path('../parsed_flows/facebook-chat.csv')
    p = Path('../pcaps/email1a.pcap')

    model, _, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = Classifier(model, device)
    # ds = BlocksDataSet.from_flows_file(f, 1)
    # a, b = c.classify_dataset(ds)
    # print(f'direct {a}')
    #
    # c = FlowCsvClassifier(model, device)
    # print('csv', Counter([b.pred for b in c.classify_file(f)[0].classified_blocks]))

    c = PcapClassifier(model, device)
    print('pcap', Counter([b.pred for b in c.classify_file(p, 3)[0].classified_blocks]))

    # ds = FlowsDataSet(file_samples, global_label=3)
    # dl = DataLoader(ds, batch_size=128, shuffle=True)
    #
    # cnt = Counter([])
    # f = 0
    # dl_iter = iter(dl)
    # for j in range(len(dl)):
    #     x, y = next(dl_iter)
    #     pred = c.classify(x)
    #     pred = pred.cpu()
    #     pred = pred.tolist()
    #     cnt += Counter(pred)
    # print('total =', len(ds))
    # print('f1 =', f/len(dl))
    # print(cnt)


