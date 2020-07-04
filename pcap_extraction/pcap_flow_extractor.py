import os
import re
from os.path import splitext
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import pyshark
import csv
import subprocess as sp


from pyshark.packet.packet import Packet
from heapq import nlargest

from flowpic_dataset.dataset import BlocksDataSet
from flowpic_dataset.processors import BasicProcessor, get_dir_items
from misc.constants import PACKET_SIZE_LIMIT
from misc.data_classes import Flow
from misc.output import Progress
from misc.constants import PROFILE, CAPINFOS_AVG_PACKET_SIZE, CAPINFOS_BIT_RATE, CAPINFOS_PACKET_COUNT


LOADING_BAR_UPDATE_INTERVAL = 500


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
                         'not dns and ' \
                         'not icmp and ' \
                         'not igmp and ' \
                         'not ntp ' \
                         'and not stun'

        pcap_metadata = self.get_pcap_metadata(file)
        packet_count = pcap_metadata[CAPINFOS_PACKET_COUNT]

        print('avg bit rate: ', pcap_metadata[CAPINFOS_BIT_RATE])
        print('avg packet size: ', pcap_metadata[CAPINFOS_AVG_PACKET_SIZE])

        capture = pyshark.FileCapture(str(file),
                                      custom_parameters={"-C": PROFILE},
                                      display_filter=display_filter,
                                      only_summaries=True)
        for i, packet in enumerate(capture):
            if i % LOADING_BAR_UPDATE_INTERVAL == 0:
                self.progress.set_counter(i, packet_count)

            packet_meta = (float(packet._fields['Time']),  int(packet._fields['Length']))
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
        capture.close()

        if n is None:
            max_five_tuples = list(packet_streams.keys())
        else:
            max_five_tuples = nlargest(n, packet_streams, key=lambda key: len(packet_streams.get(key)))

        return [self._transform_stream_to_flow(five_tuple, packet_streams[five_tuple])
                for five_tuple in max_five_tuples]

    @staticmethod
    def get_pcap_metadata(filepath):
        cmd_args = ["-c", "-z", "-i"]
        cmd_line = ["capinfos"] + cmd_args + [os.path.expanduser(str(filepath))]
        output = sp.check_output(cmd_line).decode('utf-8')
        data = re.findall(r'(.+?):\s*([\s\S]+?)(?=\n[\S]|$)', output)
        infos_dict = {i[0]: i[1] for i in data}
        for key in infos_dict:
            if 'Interface #' in key:
                iface_infos = re.findall(r'\s*(.+?) = (.+)\n', infos_dict[key])
                infos_dict[key] = {i[0]: i[1] for i in iface_infos}
        infos_dict[CAPINFOS_PACKET_COUNT] = PcapParser.parse_capinfos_packet_count(infos_dict[CAPINFOS_PACKET_COUNT])
        return infos_dict

    @staticmethod
    def parse_capinfos_packet_count(packet_count_str: str):
        if not packet_count_str.__contains__(" "):
            return int(packet_count_str)
        quantity, resolution = packet_count_str.split("\r")[0].split(" ")
        if resolution is "k":
            return int(int(quantity) * 1e3)
        elif resolution is "m":
            return int(int(quantity) * 1e6)
        elif resolution is "g":
            return int(int(quantity) * 1e9)
        return int(quantity)

    @staticmethod
    def _transform_stream_to_flow(five_tuple: Tuple, stream: Sequence[Tuple[float, int]]):
        times, sizes = zip(*stream)
        return Flow.create('app', five_tuple, times[0], times, sizes, normelize=True)

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
