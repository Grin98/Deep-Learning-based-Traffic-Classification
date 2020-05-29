from collections import Counter
from pathlib import Path

from classification.clasification import Classifier
from flowpic_dataset.dataset import FlowsDataSet
from flowpic_dataset.processors import BasicProcessor
from model.flow_pic_model import FlowPicModel
from pcap_extraction.pcap_flow_extractor import PcapParser
from misc.utils import load_model


class PcapClassifier:

    def __init__(self, model, device: str, num_categories,
                 num_flows_to_classify: int = 1,
                 seed: int = 42,
                 block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500,
                 ):
        self.num_categories = num_categories
        self.classifier = Classifier(model, device, seed)
        self.parser = PcapParser(num_flows_to_return=num_flows_to_classify)
        self.processor = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)

    def classify(self, file: Path):
        flow_rows = self.parser.parse_file(file)
        dss = [FlowsDataSet.from_flow_rows([row]) for row in flow_rows]
        flow_preds = []
        categories_distributions = []
        for i in range(self.num_categories):
            categories_distributions.append(Counter([]))

        for ds in dss:
            distribution = self.classifier.classify_dataset(ds)
            pred = distribution.most_common(1)[0][0]
            flow_preds.append(pred)
            categories_distributions[pred] += distribution

        return flow_preds, categories_distributions


if __name__ == '__main__':
    device = 'cuda'
    categories = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
    file_checkpoint = '../reg_overlap_split'
    f = Path('../pcaps/youtube2.pcap')

    model, _, _ = load_model(file_checkpoint, FlowPicModel, device)
    c = PcapClassifier(model, device, num_categories=len(categories), num_flows_to_classify=4)
    pred, dist = c.classify(f)
    print('preds', pred)
    print(categories)
    print('dist by category', dist)
