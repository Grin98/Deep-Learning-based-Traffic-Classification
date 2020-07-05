import re
import subprocess as sp
import os
from pyshark import FileCapture
from misc.constants import *


class FlowMetadata:
    def __init__(self, packet_meta):
        self.start_time = self.end_time = packet_meta[0]
        self.total_packet_data = packet_meta[1]
        self.total_packet_amount = 1

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
               f"PACKET_COUNT: {self.total_packet_amount} ({round((self.total_packet_amount / pcap_packet_count) * 100, 2)}%), " \
               f"THROUGHPUT: {self.get_throughput()} ({round((self.get_throughput() / pcap_throughput) * 100, 2)}%)"


class PcapAnalyzer:
    def __init__(self, pcap_file, prediction=None):
        self.display_filter = 'ip and ' \
                              'not dns and ' \
                              'not icmp and ' \
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

        self.flows = {flow: flow_metadata for flow, flow_metadata in
                      sorted(self.flows.items(), key=lambda entry: -entry[1].total_packet_data)}
        output = f"{pcap_meta}\n"
        for (flow, flow_metadata) in self.flows.items():
            output += f"{flow}  :  {flow_metadata.describe(pcap_meta)}\n"
        return output

    def extract_flows_map(self):
        capture = FileCapture(str(self.pcap_file),
                              custom_parameters=self.custom_parameters,
                              display_filter=self.display_filter,
                              only_summaries=True)
        for packet in capture:
            packet_meta = (float(packet._fields['Time']), int(packet._fields['Length']))
            flow = (
                packet._fields['Source'],
                packet._fields['SrcPort'],
                packet._fields['Destination'],
                packet._fields['DstPort'],
                packet._fields['Protocol']
            )

            if flow in self.flows:
                self.flows[flow].add_sample(packet_meta)
            else:
                self.flows[flow] = FlowMetadata(packet_meta)
        capture.close()

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
        elif resolution == "mbps":
            return float(float(quantity) * 1e6)
        elif resolution == "gbps":
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