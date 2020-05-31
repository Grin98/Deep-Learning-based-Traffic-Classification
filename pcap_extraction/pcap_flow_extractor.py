import os
from os.path import splitext
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import pyshark
import csv

from pyshark.packet.packet import Packet
from heapq import nlargest

from flowpic_dataset.dataset import FlowDataSet
from flowpic_dataset.processors import BasicProcessor


class PcapParser:

    def __init__(self, num_flows_to_return: int = 1):
        self.num_flows_to_return = num_flows_to_return

    def parse_file(self, file: Path) -> Sequence:
        file_extension = file.suffix
        if file_extension != '.pcap' and file_extension != '.pcapng':
            return []

        packet_streams = {}
        capture = pyshark.FileCapture(str(file), keep_packets=False)
        for packet in capture:
            if self.is_undesired_packet(packet):
                continue

            packet_meta = self.extract_packet_meta(packet)
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
        capture.close()

        max_five_tuples = nlargest(self.num_flows_to_return, packet_streams, key=lambda key: len(packet_streams.get(key)))
        print(max_five_tuples)
        return [self.transform_stream_to_flow_row(five_tuple, packet_streams[five_tuple])
                for five_tuple in max_five_tuples]

    @staticmethod
    def write_flow_rows(file: Path, flow_rows: Sequence):
        with file.open(mode='w+', newline='') as out:
            writer = csv.writer(out, delimiter=',')
            for row in flow_rows:
                writer.writerow(row)

    @staticmethod
    def transform_stream_to_flow_row(five_tuple: Tuple, stream: Sequence[Tuple[float, int]]):
        times, sizes = zip(*stream)

        # normalize time to start from 0
        start = times[0]
        times = [t - start for t in times]

        # format: app|src_ip|src_port|dst_ip|dst_port|transport_protocol|start_time|length|[timestamps]|' '|[sizes]|
        return ['app_place_holder'] + list(five_tuple) + [start] + [len(times)] + \
               times + [' '] + list(sizes)

    @staticmethod
    def is_undesired_packet(packet: Packet) -> bool:
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
    def extract_packet_meta(packet: Packet) -> Tuple[float, int]:
        ip = packet['IP'] if 'IP' in packet else packet['IPv6']
        size = int(ip.len)
        return float(packet.sniff_timestamp), size


if __name__ == '__main__':
    file = Path('../pcaps/facebook-chat.pcapng')
    parser = PcapParser(num_flows_to_return=2)
    flow_rows = parser.parse_file(file)
    dss = [FlowDataSet.from_flows([row]) for row in flow_rows]
    for ds in dss:
        print(len(ds))
        print(ds.data)
        print(ds.start_times)
