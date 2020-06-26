from pyshark import LiveCapture
from pyshark.packet.packet import Packet
from collections import deque
from typing import Tuple
import os


class NonPromiscuousLiveCapture(LiveCapture):
    def get_parameters(self, packet_count=None):
        """
        Returns the tshark parameters to be used according to the configuration of this class.
        The -p flag is added to disable promiscuous mode
        """
        params = super(LiveCapture, self).get_parameters(packet_count=packet_count)
        for interface in self.interfaces:
            params += ['-i', interface]
        if self.bpf_filter:
            params += ['-f', self.bpf_filter]
        params += ['-p']
        return params


class FlowData:
    def __init__(self, sample):
        self.samples = [sample]
        # this is a size 4 array with the amount of samples from that flow during the last: [0,15], [15,30], [30,
        # 45] and [45,60] seconds respectively
        self.window_frequencies = [1, 0, 0, 0]

    def add_sample(self, sample):
        self.samples.append(sample)
        self.window_frequencies[0] += 1

    def advance_sliding_window(self):
        self.window_frequencies[3] = self.window_frequencies[2]
        self.window_frequencies[2] = self.window_frequencies[1]
        self.window_frequencies[1] = self.window_frequencies[0]
        self.window_frequencies[0] = 0


class FlowsManager:
    """
    Unified class for all sorts of interactions on flows' monitoring during live capture
    """

    def __init__(self):
        self.flows = {}

    def advance_sliding_window(self):
        for _, flow_data in self.flows:
            flow_data.advance_sliding_window()

    def add_sample(self, flow, sample):
        if flow in self.flows:
            self.flows[flow].add_sample(sample)
        else:
            self.flows[flow] = FlowData(sample)


class LiveCaptureProvider:
    """
    Performs a live capture.
    Creates a new flow after 60 seconds and every 15 seconds thereafter, and adds the flows to [queue].
    Each time a flow is created - all subscribers are notified

    :raises [exception] if no interfaces are found
            [exception] if no packets are found
    """

    def __init__(self):
        # only_summaries flag is important! contains packet size and arrival time (relative to capture start)
        # TODO: add more capture filters
        capture_filter = 'not arp and ' \
                         'port not 53 and ' \
                         'not broadcast and ' \
                         'not ip6'

        self.capture = NonPromiscuousLiveCapture(capture_filter=capture_filter)
        self.queue = deque()
        self.absolute_start_time = None
        self.relative_time = 0.0
        self.flows_manager = FlowsManager()

    def packet_callback(self, packet):
        if self.absolute_start_time is None:
            self.absolute_start_time = float(packet.sniff_timestamp)
        self.relative_time = float(packet.sniff_timestamp) - self.absolute_start_time
        transport = packet.transport_layer
        flow = (
            packet['ip'].src,
            packet[transport].srcport,
            packet['ip'].dst,
            packet[transport].dstport,
            transport
        )
        self.flows_manager.add_sample(flow, self._extract_packet_meta(packet))
        print(self.relative_time)

    @staticmethod
    def _extract_packet_meta(packet: Packet) -> Tuple[float, int]:
        ip = packet['IP'] if 'IP' in packet else packet['IPv6']
        size = int(ip.len)
        return float(packet.sniff_timestamp), size

    def start_capture(self):
        # self.capture.load_packets(timeout=10)
        # print(self.capture)
        # for packet in self.capture:
        #     print(packet)
        self.capture.apply_on_packets(self.packet_callback)

    def stop_capture(self):
        self.capture.close()


if __name__ == '__main__':
    live = LiveCaptureProvider()
    live.start_capture()
    # live.stop_capture()
