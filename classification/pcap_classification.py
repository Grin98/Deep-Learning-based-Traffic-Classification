from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple

from classification.clasification import Classifier
from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.processors import BasicProcessor
from misc.data_classes import ClassifiedFlow
from model.flow_pic_model import FlowPicModel
from pcap_extraction.pcap_flow_extractor import PcapParser
from misc.utils import load_model, set_seed


class PcapClassifier:

    def __init__(self, model, device: str, num_categories,
                 seed: int = 42,
                 packet_size_limit: int = 1500,
                 ):
        """

        :param model: model that will be used for classifying FlowPics
        :param device: if value is cuda then the model will run on gpu and if cpu then it will run on cpu
        :param num_categories: the number of possible different classifications
        :param seed: seed for numpy and torch
        :param packet_size_limit: size in bytes where larger packages will be discarded
        """
        set_seed(seed)
        self.num_categories = num_categories
        self.packet_size_limit = packet_size_limit
        self.classifier = Classifier(model, device, seed)
        self.parser = PcapParser()

    def classify_file(self, file: Path, num_flows_to_classify: int) -> Sequence[ClassifiedFlow]:
        print(f'parsing file {str(file)}')
        flows = self.parser.parse_file(file, num_flows_to_classify, self.packet_size_limit)
        return self.classifier.classify_multiple_flows(flows)

    def classify_multiple_files(self, files: Sequence[Path], num_flows_to_classify: int = 1) -> Sequence[ClassifiedFlow]:
        classified_flows = list(itertools.chain.from_iterable([self.classify_file(f, num_flows_to_classify)
                                                               for f in files]))
        return classified_flows


if __name__ == '__main__':
    device = 'cuda'
    categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
    file_checkpoint = '../reg_overlap_split'
    f1 = Path('../pcaps/youtube2.pcap')
    f2 = Path('../pcaps/facebook-chat.pcapng')

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = PcapClassifier(model, device, num_categories=len(categories))
    a, b = c.classify_multiple_files([f2, f1], num_flows_to_classify=1)
    print(a.pd)
    # print([(f.pred, f.flow.five_tuple) for f in b])
    # a, b = c.classify_file(f1, num_flows_to_classify=1)
    # print(a.pd)
    # a, b = c.classify_file(f2, num_flows_to_classify=1)
    # print(a.pd)
    # print('preds', pred)
    # print(categories)
    # print('dist by category', dist)
