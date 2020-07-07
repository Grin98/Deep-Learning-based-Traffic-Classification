import re
import subprocess as sp
import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Sequence

from pyshark import FileCapture
from misc.constants import *
from misc.data_classes import Flow
from misc.utils import write_flows
from pcap_extraction.pcap_flow_extractor import PcapParser

MINIMUM_FLOW_PACKETS = 5


class FlowMetadata:
    def __init__(self, packet_meta):
        self.start_time = self.end_time = packet_meta[0]
        self.total_packet_data = packet_meta[1]
        self.total_packet_amount = 1
        self.flow = None

    @classmethod
    def create(cls, flow: Flow):
        fmd = FlowMetadata((0, 0))
        fmd.flow = flow
        fmd.start_time = flow.start_time
        fmd.end_time = flow.times[-1] + flow.start_time
        fmd.total_packet_data = sum(flow.sizes)
        fmd.total_packet_amount = len(flow.times)
        return fmd

    def add_sample(self, packet_meta):
        self.end_time = packet_meta[0]
        self.total_packet_data += packet_meta[1]
        self.total_packet_amount += 1

    def get_time_alive(self):
        return round(self.end_time - self.start_time, 2)

    def get_throughput(self):
        if self.get_time_alive() == 0:
            return 0
        return round(self.total_packet_data * BYTES_IN_BITS / self.get_time_alive(), 2)

    def get_metadata(self):
        return self.get_time_alive(), self.total_packet_amount, self.get_throughput()

    def __str__(self):
        return f"TIME ALIVE: {self.get_time_alive()}, " \
            f"PACKET_COUNT: {self.total_packet_amount}, " \
            f"THROUGHPUT: {self.get_throughput()}"

    def describe(self, pcap_meta):
        pcap_packet_count, pcap_throughput, pcap_avg_packet_size, pcap_capture_duration = \
            pcap_meta[CAPINFOS_PACKET_COUNT], pcap_meta[CAPINFOS_BIT_RATE], \
            pcap_meta[CAPINFOS_AVG_PACKET_SIZE], pcap_meta[CAPINFOS_CAPTURE_DURATION]
        return f"TIME ALIVE: {self.get_time_alive()} ({round((self.get_time_alive() / pcap_capture_duration) * 100, 2)}%), " \
            f"PACKET_COUNT: {self.total_packet_amount} ({round((self.total_packet_amount / pcap_packet_count) * 100,2)}%), " \
            f"THROUGHPUT: {self.get_throughput()} ({round((self.get_throughput() / pcap_throughput) * 100, 2)}%)"


class PcapAnalyzer:
    def __init__(self, pcap_file: Path, prediction=None):
        self.display_filter = 'ip and ' \
                         'not ipv6 and ' \
                         'not dns and ' \
                         'not icmp and ' \
                         'not icmpv6 and ' \
                         'not igmp and ' \
                         'not ntp ' \
                         'and not stun'

        self.custom_parameters = {"-C": PROFILE}
        self.prediction = prediction
        self.pcap_file = pcap_file
        self.flows = {}

    def set_prediction(self, prediction):
        self.prediction = prediction

    def analyze(self):
        self.flows.clear()
        self.extract_flows_map()
        pcap_meta = self.get_pcap_metadata(self.pcap_file)

        self.flows = {flow: self.flows[flow] for flow in self.flows if
                      self.flows[flow].total_packet_amount > MINIMUM_FLOW_PACKETS}
        self.flows = OrderedDict(
            sorted(self.flows.items(), key=lambda entry: entry[1].total_packet_amount, reverse=True))
        output = f"{pcap_meta}\n"
        for (flow, flow_metadata) in self.flows.items():
            output += f"{flow}  :  {flow_metadata.describe(pcap_meta)}\n"
        return output

    def extract_flows_map(self):
        parser = PcapParser()
        flows = parser.parse_file(self.pcap_file)
        for flow in flows:
            self.flows[str(flow.five_tuple)] = FlowMetadata.create(flow)

    def write_chosen_flows(self, writable, flow_indices: Sequence[int], labels: str):
        if len(flow_indices) != len(set(flow_indices)):
            raise Exception("there are duplicate indices")

        flows = list(self.flows.values())
        flows = [Flow.change_app(flows[i - 1].flow, label)
                 for i, label in zip(flow_indices, labels)]

        write_flows(writable, flows)

    @staticmethod
    def get_pcap_metadata(filepath):
        cmd_args = ["-c", "-z", "-i", "-u"]
        cmd_line = ["capinfos"] + cmd_args + [os.path.expanduser(str(filepath))]
        output = sp.check_output(cmd_line).decode('utf-8')
        data = re.findall(r'(.+?):\s*([\s\S]+?)(?=\n[\S]|$)', output)
        infos_dict = {i[0]: i[1] for i in data}
        for key in infos_dict:
            if 'Interface #' in key:
                iface_infos = re.findall(r'\s*(.+?) = (.+)\n', infos_dict[key])
                infos_dict[key] = {i[0]: i[1] for i in iface_infos}
        return PcapAnalyzer.parse_capinfos_dict(infos_dict)

    @staticmethod
    def parse_capinfos_packet_count(packet_count_str: str):
        if " " not in packet_count_str:
            return int(packet_count_str)
        quantity, resolution = packet_count_str.split("\r")[0].split(" ")
        if resolution == "k":
            return int(int(quantity) * 1e3)
        elif resolution == "m":
            return int(int(quantity) * 1e6)
        elif resolution == "g":
            return int(int(quantity) * 1e9)
        return int(quantity)

    @staticmethod
    def parse_capinfos_capture_duration(capture_duration_str: str):
        return float(capture_duration_str.split(" ")[0])

    @staticmethod
    def parse_capinfos_bit_rate(capture_bit_rate_str: str):
        quantity, resolution = capture_bit_rate_str.split("\r")[0].split(" ")
        if resolution == "kbps":
            return float(float(quantity) * 1e3)
        elif resolution == "Mbps":
            return float(float(quantity) * 1e6)
        elif resolution == "Gbps":
            return float(float(quantity) * 1e9)
        return float(quantity)

    @staticmethod
    def parse_capinfos_avg_packet_size(capture_avg_packet_size_str: str):
        return float(capture_avg_packet_size_str.split("\r")[0].split(" ")[0])

    @staticmethod
    def parse_capinfos_dict(infos_dict):
        infos_dict[CAPINFOS_PACKET_COUNT] = PcapAnalyzer.parse_capinfos_packet_count(infos_dict[CAPINFOS_PACKET_COUNT])
        infos_dict[CAPINFOS_CAPTURE_DURATION] = \
            PcapAnalyzer.parse_capinfos_capture_duration(infos_dict[CAPINFOS_CAPTURE_DURATION])
        infos_dict[CAPINFOS_BIT_RATE] = PcapAnalyzer.parse_capinfos_bit_rate(infos_dict[CAPINFOS_BIT_RATE])
        infos_dict[CAPINFOS_AVG_PACKET_SIZE] = \
            PcapAnalyzer.parse_capinfos_avg_packet_size(infos_dict[CAPINFOS_AVG_PACKET_SIZE])
        return infos_dict


if __name__ == '__main__':
    analyzer = PcapAnalyzer(r"C:\Users\Sahar\Desktop\Code\Networking Traffic Classification\pcaps\ms_recording.pcapng")
    print(analyzer.analyze())
