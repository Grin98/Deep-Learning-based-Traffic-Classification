import csv
from pathlib import Path
from typing import Sequence, Iterable

from misc.data_classes import Flow
from misc.output import Progress
from misc.utils import write_flows
from pcap_extraction.pcap_flow_extractor import PcapParser


d = dict(
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
    youtube='video',
    html='browsing'
)


class PcapAggregator:
    def __init__(self, progress=Progress()):
        self.progress = progress

    def aggregate(self, out_file: Path, pcaps: Sequence[Path], ns: Sequence[int],
                  labels):
        """
        :param out_file: the path to where the file is (file will be created if it doesn't exists)
        :param pcaps: paths to the pcap files that will be aggregated together into a file
        :param ns: how many flows to take from each pcap (takes the top n largest flows)
        :param labels: the classification of the flows either  at the file level or the flow level
        """
        parser = PcapParser(self.progress)
        with out_file.open(newline='', mode='w+') as f:
            writer = csv.writer(f, delimiter=',')

            for pcap, n, label in zip(pcaps, ns, labels):
                flows = parser.parse_file(pcap, n)
                flows = self._label_flows(flows, label)
                write_flows(writer, flows)

    @staticmethod
    def _label_flows(flows, label):
        apps = label if (isinstance(label, list) or isinstance(label, tuple)) else [label] * len(flows)
        print(apps)
        if len(apps) != len(flows):
            raise Exception("number of labels isn't equal to the number of flows")

        return [Flow.change_app(f, app) for f, app in zip(flows, apps)]
