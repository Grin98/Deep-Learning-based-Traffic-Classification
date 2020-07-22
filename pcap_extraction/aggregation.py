import csv
import numpy as np
from pathlib import Path
from typing import Sequence

from flowpic_dataset.processors import BasicProcessor
from misc.data_classes import Flow
from misc.output import Progress
from misc.utils import write_flows
from pcap_extraction.pcap_flow_extractor import PcapParser


class Aggregator:
    def __init__(self, progress=Progress()):
        self.progress = progress
        self.parser = PcapParser(self.progress)
        self.processor = BasicProcessor()

    def aggregate_pcaps(self, out_file: Path, pcaps: Sequence[Path], ns: Sequence[int],
                        labels):
        """
        extracts flows from pcaps.
        attaches classification to the flows.
        writes them to a file.

        :param out_file: the path to where the file is (file will be created if it doesn't exists)
        :param pcaps: paths to the pcap files that will be aggregated together into a file
        :param ns: how many flows to take from each pcap (takes the top n largest flows)
        :param labels: the classification of the flows either  at the file level or the flow level
        """
        with out_file.open(newline='', mode='w+') as f:
            writer = csv.writer(f, delimiter=',')

            for pcap, n, label in zip(pcaps, ns, labels):
                print(pcap.name)
                self.write_pcap_flows(writer, pcap, n, label)

    def write_pcap_flows(self, writable, pcap: Path, n: int, labels):
        """
        extracts flows from pcap.
        attaches classification to the flows.
        writes them to a file.

        :param writable: a file or an object with writerows method
        :param pcap: pcap file path
        :param n: num flows to extract
        :param labels: flow labels either a string or sequence of strings
        """

        flows = self.parser.parse_file(pcap, n)
        flows = self._label_flows(flows, labels)
        write_flows(writable, flows)

    def merge_csvs(self, out_file: Path, csvs: Sequence[Path], random_start: bool = True):
        """
        writes flows from different csv files to a single csv
        :param out_file: file to write to
        :param csvs: csv files with flows
        :param random_start: if true csv files start from random value.
                explanation: flows in csv are assumed to come from the same pcap file where
                the start of each flow is relative to the start of the pcap (pcap start is 0)
                thus if random_start is True, the start time of each csv is moved from 0 to
                a random value between 0 and the maximum time of when a flow ends.
        """

        with out_file.open(newline='', mode='w+') as f:
            writer = csv.writer(f, delimiter=',')

            if not random_start:
                for file in csvs:
                    flows = self.processor.process_file_to_flows(file)
                    write_flows(writer, flows)
            else:
                end_times = [self._get_file_end_time(file) for file in csvs]
                max_end = max(end_times)
                for file, end in zip(csvs, end_times):
                    flows = self.processor.process_file_to_flows(file)
                    new_pcap_start_time = np.random.uniform(low=0.0, high=max_end - end)
                    flows = [Flow.change_start_time(f, new_pcap_start_time + f.start_time) for f in flows]
                    write_flows(writer, flows)

    def _get_file_end_time(self, file: Path) -> float:
        """
        returns the maximum time of when a flow ends in the given file
        """
        flows = self.processor.process_file_to_flows(file)
        return max([f.times[-1] for f in flows])

    @staticmethod
    def _label_flows(flows, label):
        """
        attaches labels to the flows
        """
        apps = label if (isinstance(label, list) or isinstance(label, tuple)) else [label] * len(flows)
        if len(apps) != len(flows):
            raise Exception("number of labels isn't equal to the number of flows")

        return [Flow.change_app(f, app) for f, app in zip(flows, apps)]
