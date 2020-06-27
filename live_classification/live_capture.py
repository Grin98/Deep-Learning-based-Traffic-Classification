from pyshark import LiveCapture
from pyshark.packet.packet import Packet
from collections import deque
from typing import Tuple
from misc.data_classes import Block
import itertools

BLOCK_LENGTH = 60
BLOCK_INTERVAL = 15


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
    def __init__(self, sample, absolute_start_time):
        # these are windows with the samples from the flow during the last: [0,15], [15,30], [30,45] and [45,60]
        # seconds. self.head refers to the "most recent" window, and every subsequent number modulo 4 will be the next
        self.window_samples = [[] for _ in range(BLOCK_LENGTH // BLOCK_INTERVAL)]
        self.head = 0
        self.window_samples[self.head].append(sample)
        self.absolute_start_time = absolute_start_time

    def add_sample(self, sample):
        self.window_samples[self.head].append(sample)

    def advance_sliding_window(self):
        self.window_samples[(self.head + len(self.window_samples) - 1) % len(self.window_samples)].clear()
        self.head = (self.head - 1) % len(self.window_samples)
        self.absolute_start_time += BLOCK_INTERVAL
        return self

    def __len__(self):
        return sum(len(window) for window in self.window_samples)

    def to_block(self):
        return Block(start_time=self.absolute_start_time, num_packets=self.__len__(),
                     data=list(itertools.chain.from_iterable(window for window in self.window_samples)))


class FlowsManager:
    """
    Unified class for all sorts of interactions on flows' monitoring during live capture
    """

    def __init__(self):
        self.flows = {}

    def advance_sliding_window(self):
        self.flows = {flow: flow_data.advance_sliding_window()
                      for flow, flow_data in self.flows.items()
                      if len(flow_data) is not 0}

    def add_sample(self, flow, sample, absolute_timestamp):
        if flow in self.flows:
            self.flows[flow].add_sample(sample)
        else:
            self.flows[flow] = FlowData(sample, absolute_start_time=absolute_timestamp)

    def compose_new_batch(self):
        batch = []
        for flow, flow_data in self.flows.items():
            batch.append((flow, flow_data.to_block()))
        return batch


class LiveCaptureProvider:
    """
    Performs a live capture.
    Creates a new block after 60 seconds and every 15 seconds thereafter, and adds the flows to [queue].
    Each time a flow is created - all subscribers are notified

    :raises [exception] if no interfaces are found
            [exception] if no packets are found
    """

    def __init__(self):
        # only_summaries flag is important! contains packet size and arrival time (relative to capture start)
        # TODO: see if we need to add more capture filters
        capture_filter = 'not arp and ' \
                         'port not 53 and ' \
                         'not broadcast and ' \
                         'not ip6 and ' \
                         'not igmp and ' \
                         'not icmp and ' \
                         'port not 123'

        self.capture = NonPromiscuousLiveCapture(capture_filter=capture_filter)
        self.queue = deque()
        self.absolute_start_time = None
        self.relative_time = 0.0
        self.sliding_window_advancements = 0
        self.flows_manager = FlowsManager()

    def packet_callback(self, packet):
        if self.absolute_start_time is None:
            self.absolute_start_time = float(packet.sniff_timestamp)
        self.relative_time = float(packet.sniff_timestamp) - self.absolute_start_time

        self.advance_sliding_window_if_needed()

        transport = packet.transport_layer
        flow = (
            packet['ip'].src,
            packet[transport].srcport,
            packet['ip'].dst,
            packet[transport].dstport,
            transport
        )
        self.flows_manager.add_sample(flow, self._extract_packet_meta(packet),
                                      self.relative_time + self.absolute_start_time)

    def advance_sliding_window_if_needed(self):
        while self.relative_time >= (self.sliding_window_advancements + 1) * BLOCK_INTERVAL:
            self.sliding_window_advancements += 1
            if self.sliding_window_advancements >= BLOCK_LENGTH // BLOCK_INTERVAL:
                self.push_flows()
                self.flows_manager.advance_sliding_window()

    def push_flows(self):
        self.queue.append(self.flows_manager.compose_new_batch())
        # print('created new block at time:', self.relative_time)
        # for item in self.queue:
        #     print(item)
        # self.queue.clear()

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
