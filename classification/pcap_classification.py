from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple

from classification.clasification import Classifier
from flowpic_dataset.dataset import FlowDataSet
from flowpic_dataset.processors import BasicProcessor
from misc.data_classes import ClassifiedFlow
from model.flow_pic_model import FlowPicModel
from pcap_extraction.pcap_flow_extractor import PcapParser
from misc.utils import load_model, fix_seed


class PcapClassifier:

    def __init__(self, model, device: str, num_categories,
                 seed: int = 42,
                 block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500,
                 ):
        fix_seed(seed)
        self.num_categories = num_categories
        self.packet_size_limit = packet_size_limit
        self.classifier = Classifier(model, device, seed)
        self.parser = PcapParser()
        self.processor = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)

    def classify_file(self, file: Path, num_flows_to_classify: int = 1) -> Tuple[PredDist, Sequence[ClassifiedFlow]]:
        print('parsing file')
        flows = self.parser.parse_file(file, num_flows_to_classify, self.packet_size_limit)
        dss = [FlowDataSet.from_flows([f]) for f in flows]

        classified_dist = PredDist(self.num_categories)
        flow_preds = []
        classified_flows = []
        for i, ds in enumerate(dss):
            print('classifying %s' % str(flows[i].five_tuple))
            distribution, classified_blocks = self.classifier.classify_dataset(ds)
            pred = distribution.most_common(1)[0][0]
            flow_preds.append(pred)
            classified_dist.update_pred(pred, distribution)
            classified_flows += [ClassifiedFlow(flows[i], pred, classified_blocks)]

        return classified_dist, classified_flows

    def classify_multiple_files(self, files: Sequence[Path], num_flows_to_classify: int = 1) -> Tuple[PredDist, Sequence[ClassifiedFlow]]:
        classified_dist = PredDist(self.num_categories)
        classified_flows = []
        for f in files:
            print('classifying %s' % str(f))
            cd, cf = self.classify_file(f, num_flows_to_classify)
            classified_dist.join_with(cd)
            classified_flows += cf

        return classified_dist, classified_flows


class PredDist:
    def __init__(self, num_preds):
        self.n = num_preds
        self.pd = [Counter([]) for _ in range(num_preds)]

    def update_pred(self, pred: int, c: Counter):
        self.pd[pred] += c
        return self

    def join_with(self, other: PredDist):
        if self.n != other.n:
            raise Exception('self and other have different number of possible preds')
        self.pd = [self.pd[i] + other.pd[i] for i in range(self.n)]
        return self

if __name__ == '__main__':
    device = 'cuda'
    categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
    file_checkpoint = '../reg_overlap_split'
    f1 = Path('../pcaps/youtube2.pcap')
    f2 = Path('../pcaps/facebook-chat.pcapng')

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = PcapClassifier(model, device, num_categories=len(categories))
    # a, b = c.classify_multiple_files([f2, f1], num_flows_to_classify=2)
    # print(a.pd)
    # print([(f.pred, f.flow.five_tuple) for f in b])
    a, b = c.classify_file(f2, num_flows_to_classify=1)
    print(a.pd)
    a, b = c.classify_file(f1, num_flows_to_classify=1)
    print(a.pd)
    # print('preds', pred)
    # print(categories)
    # print('dist by category', dist)
