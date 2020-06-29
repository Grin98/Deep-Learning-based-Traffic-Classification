import os
from os.path import splitext
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import pyshark
import csv

from pyshark.packet.packet import Packet
from heapq import nlargest

from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.processors import BasicProcessor, get_dir_items
from misc.constants import PACKET_SIZE_LIMIT
from misc.data_classes import Flow
from misc.output import Progress
from misc.constants import PROFILE


class StatisticalDataFileCapture(pyshark.FileCapture):
    def get_parameters(self, packet_count=None):
        params = super(pyshark.FileCapture, self).get_parameters(packet_count=packet_count)
        params += ['-C', PROFILE]
        return params


class PcapParser:

    def __init__(self, progress=Progress()):
        self.progress = progress

    def parse_file(self, file: Path, n: int = None) -> Sequence[Flow]:
        """
        returns flows from a pcap file
        :param file: the pcap file to be parsed, either .pcap or .pcapng
        :param n: the number of flows to be returned
        :param packet_size_limit: size in bytes where larger packages will be discarded
        :return: the top n flows with the most number of pcaps in the pcap
        """
        file_extension = file.suffix
        if file_extension != '.pcap' and file_extension != '.pcapng':
            return []

        self.progress.counter_title('parsed packets').set_counter(0)
        packet_streams = {}
        display_filter = 'ip and ' \
                         'udp.port != 53 and ' \
                         'not ipv6 and ' \
                         'not igmp and ' \
                         'not icmp and ' \
                         'udp.port != 123'
        capture = StatisticalDataFileCapture(str(file),
                                             keep_packets=True,
                                             only_summaries=True,
                                             display_filter=display_filter)
        for i, packet in enumerate(capture):
            if i % 500 == 0:
                self.progress.set_counter(i)

            packet_meta = ( packet._fields['Time'],  packet._fields['Length'])
            five_tuple = (
                packet._fields['Source'],
                packet._fields['SrcPort'],
                packet._fields['Destination'],
                packet._fields['DstPort'],
                packet._fields['Protocol']
            )

            if five_tuple in packet_streams:
                packet_streams[five_tuple].append(packet_meta)
            else:
                packet_streams[five_tuple] = [packet_meta]
        pcap_start_time = float(capture[0]._fields['Time'])
        capture.close()

        if n is None:
            max_five_tuples = list(packet_streams.keys())
        else:
            max_five_tuples = nlargest(n, packet_streams, key=lambda key: len(packet_streams.get(key)))

        return [self._transform_stream_to_flow(five_tuple, packet_streams[five_tuple], pcap_start_time)
                for five_tuple in max_five_tuples]

    @staticmethod
    def _transform_stream_to_flow(five_tuple: Tuple, stream: Sequence[Tuple[float, int]], pcap_start_time: float):
        times, sizes = zip(*stream)
        pcap_relative_start_time = times[0] - pcap_start_time
        return Flow.create('app', five_tuple, times, sizes, pcap_relative_start_time=pcap_relative_start_time)

    @staticmethod
    def write_flow_rows(file: Path, flows: Sequence[Flow]):
        with file.open(mode='w+', newline='') as out:
            writer = csv.writer(out, delimiter=',')
            for f in flows:
                writer.writerow(f.convert_to_row())

    @staticmethod
    def _is_undesired_packet(packet: Packet) -> bool:
        if packet.transport_layer is None:
            return True
        if 'IPv6' in packet:
            # for now, we ignore all IPv6 packets
            return True
        elif 'ICMPv6' in packet:
            return True
        elif 'DNS' in packet:
            return True
        elif 'STUN' in packet:
            return True
        elif 'NTS' in packet:
            return True

        return False

    @staticmethod
    def _extract_packet_meta(packet: Packet) -> Tuple[float, int]:
        ip = packet['IP'] if 'IP' in packet else packet['IPv6']
        size = int(ip.len)
        return float(packet.sniff_timestamp), size


if __name__ == '__main__':
    # pcap = Path('../pcaps/scpDown2.pcap')
    # csv = Path('../parsed_flows/scpDown2.csv')
    # flows = PcapParser().parse_file(pcap, n=1)
    # flows = BasicProcessor().process_file_to_flows(csv)
    # print(flows[0].num_packets, flows[0].times, flows[0].sizes)
    # exit()
    d = Path('../pcaps')
    parser = PcapParser()
    t = len(get_dir_items(d))
    for i, f in enumerate(get_dir_items(d)):
        print(f'{i} / {t}')
        flows = parser.parse_file(f, n=1)
        parser.write_flow_rows(Path(f'../parsed_flows/{f.stem}.csv'), flows)

    # dss = [BlocksDataSet.from_flows([f]) for f in flows]
    # for ds in dss:
    #     print(len(ds))
    #     print(ds.)
    #     print(ds.start_times)
