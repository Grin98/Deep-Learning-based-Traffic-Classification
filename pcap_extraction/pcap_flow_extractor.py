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
from flowpic_dataset.processors import BasicProcessor
from misc.constatns import PACKET_SIZE_LIMIT
from misc.data_classes import Flow


class PcapParser:

    def parse_file(self, file: Path, n: int) -> Sequence[Flow]:
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

        packet_streams = {}
        capture = pyshark.FileCapture(str(file), keep_packets=True)
        for packet in capture:
            if self._is_undesired_packet(packet):
                continue

            packet_meta = self._extract_packet_meta(packet)
            transport = packet.transport_layer
            five_tuple = (
                packet.ip.src,
                packet[transport].srcport,
                packet.ip.dst,
                packet[transport].dstport,
                transport
            )

            if five_tuple in packet_streams:
                packet_streams[five_tuple].append(packet_meta)
            else:
                packet_streams[five_tuple] = [packet_meta]
        pcap_start_time = float(capture[0].sniff_timestamp)
        capture.close()

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
    file = Path('../pcaps/facebook-chat.pcapng')
    parser = PcapParser()
    flows = parser.parse_file(file, n=1)
    f = flows[0]
    print(f.five_tuple)
    print(f.start_time, f.pcap_relative_start_time, f.num_packets)
    print(f.times)
    # dss = [BlocksDataSet.from_flows([f]) for f in flows]
    # for ds in dss:
    #     print(len(ds))
    #     print(ds.)
    #     print(ds.start_times)
