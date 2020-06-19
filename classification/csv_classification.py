import csv
import itertools
from pathlib import Path
from typing import Sequence

from classification.clasification import Classifier
from misc.data_classes import ClassifiedFlow, Flow
from misc.utils import set_seed


class FlowCsvClassifier:
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

    def classify_file(self, file: Path) -> Sequence[ClassifiedFlow]:
        print(f'parsing file {str(file)}')
        with file.open(newline='', mode='r') as f:
            reader = csv.reader(f, delimiter=',')
            flows = [Flow.create_from_row(row, self.packet_size_limit) for row in reader]
            return self.classifier.classify_multiple_flows(flows)

    def classify_multiple_files(self, files: Sequence[Path]) -> Sequence[ClassifiedFlow]:
        classified_flows = list(itertools.chain.from_iterable([self.classify_file(f)
                                                               for f in files]))
        return classified_flows
