import csv
import numpy as np
from pathlib import Path
from typing import Sequence, Iterable, Tuple

from flowpic_dataset.processors import BasicProcessor
from misc.data_classes import Flow
from misc.output import Progress
from misc.utils import write_flows, get_dir_items, get_dir_pcaps
from pcap_extraction.pcap_flow_extractor import PcapParser

d = dict(
    html='browsing',
    chat='chat',
    video='video',
    voip='voip',
    file='file_transfer',
    netflix='video',
    audio='voip',
    ftps='file_transfer',
    scp='file_transfer',
    sftp='file_transfer',
    spotify='voip',
    torrent='file_transfer',
    vimeo='video',
    youtube='video'
)


class PcapAggregator:
    def __init__(self, progress=Progress()):
        self.progress = progress
        self.parser = PcapParser(self.progress)
        self.processor = BasicProcessor()


    def aggregate(self, out_file: Path, pcaps: Sequence[Path], ns: Sequence[int],
                  labels):
        """
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

    def write_pcap_flows(self, writable, pcap: Path, n: int, label):
        flows = self.parser.parse_file(pcap, n)
        flows = self._label_flows(flows, label)
        write_flows(writable, flows)

    def merge_csvs(self, out_file: Path, csvs: Sequence[Path], random_start: bool = False):
        print(str(out_file))
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
                    print(new_pcap_start_time)
                    print([(f.five_tuple, f.start_time) for f in flows])
                    write_flows(writer, flows)

    def _get_file_end_time(self, file: Path) -> float:
        flows = self.processor.process_file_to_flows(file)
        return max([f.times[-1] for f in flows])

    @staticmethod
    def get_file_label(file: Path):
        gb = 1e9
        if (file.stat().st_size / gb) > 1:
            print(f"{file.stat().st_size / gb:.2f}GB is too large {file}")
            return None

        name = file.stem.lower()
        for key in d.keys():
            if key in name:
                return d[key]

        print(f"don't know how to classify {file}")
        return None

    @staticmethod
    def _label_flows(flows, label):
        apps = label if (isinstance(label, list) or isinstance(label, tuple)) else [label] * len(flows)
        if len(apps) != len(flows):
            raise Exception("number of labels isn't equal to the number of flows")

        return [Flow.change_app(f, app) for f, app in zip(flows, apps)]


if __name__ == '__main__':
    files = get_dir_pcaps(Path('../../PCAPS'))
    a = PcapAggregator()
    files, labels = zip(*[(file, a.get_file_label(file)) for file in files if a.get_file_label(file) is not None])
    ns = [1] * len(files)
    out = Path('out.csv')
    a.aggregate(out, files, ns, labels)
