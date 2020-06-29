from pyshark import LiveCapture
from pyshark.packet.packet import Packet
from collections import deque
from typing import Tuple
from misc.data_classes import Block
from misc.constants import PROFILE
import itertools
import datetime
import logging

BLOCK_LENGTH = 12
BLOCK_INTERVAL = 3
# TODO: CHANGE LOGGING LEVEL FROM [logging.DEBUG] TO [logging.INFO] TO STOP DEBUG MESSAGES!!!
# logging.basicConfig(filename='example_live_capture.txt', format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


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
        params += ['-C', PROFILE]
        return params


def _anti_clockwise_crange(start, end, modulo):
    """
    A generator for an anti-clockwise cyclic range
    :param start: number of steps anti-clockwise from 0 to start at
    :param end: number of steps anti-clockwise from 0 to end at
    :param modulo: the number of steps in the cycle
    """
    for i in range(end - start):
        yield (modulo - start + i) % modulo


class FlowData:
    """
    Data class that represents an ongoing flow's data stream, divided into 4 windows.
    The class handles sliding window shifts very quickly by maintaining a cyclic ring buffer.
    """

    def __init__(self, sample, absolute_start_time):
        """
        :param sample: a 2-tuple of (arrival_time, packet_size)
        :param absolute_start_time: timestamp of the first packet in the flow
        """
        # these are windows with the samples from the flow during the last: [0,15], [15,30], [30,45] and [45,60]
        # seconds. self.head refers to the "most recent" window, and every *previous* index modulo 4 will be the next
        self.window_samples = [[] for _ in range(BLOCK_LENGTH // BLOCK_INTERVAL)]
        self.head = 0
        self.window_samples[self.head].append(sample)
        self.absolute_start_time = absolute_start_time
        self.alive_intervals = 0

    def add_sample(self, sample):
        """
        Adds [sample] to the most recent window in the buffer (dictated by self.head)
        """
        self.window_samples[self.head].append(sample)

    def advance_sliding_window(self):
        """
        Advances the sliding window.
        This dictates an anti-clockwise shift in the ring buffer, which is represented by moving self.head
        one step anti-clockwise, modulo #windows.
        The start time of the flow remains fixed, but the [self.alive_intervals] counter is incremented to later keep
        track of how much relative time to add to the start of this flow
        :return: reference to self, after shifting window
        """
        self.window_samples[(self.head + len(self.window_samples) - 1) % len(self.window_samples)].clear()
        self.head = (self.head - 1) % len(self.window_samples)
        self.alive_intervals += 1
        return self

    def __len__(self):
        return sum(len(window) for window in self.window_samples)

    def to_block(self):
        """
        :return: a 2-tuple of (flow 5-tuple, block of last 60 secs)
        """
        return Block(start_time=self.absolute_start_time + self.alive_intervals * BLOCK_INTERVAL,
                     num_packets=self.__len__(),
                     data=list(itertools.chain.from_iterable(self.window_samples[i] for i in _anti_clockwise_crange(
                         self.head, self.head + len(self.window_samples), len(self.window_samples)))))


class FlowsManager:
    """
    Unified class for all sorts of interactions on flows' monitoring during live capture
    """

    def __init__(self):
        self.flows = {}

    def advance_sliding_window(self):
        """
        Advances the sliding window for all flows. If any of the flows has no more data in it in the last 60 seconds -
        it is also deleted
        """
        self.flows = {flow: flow_data.advance_sliding_window()
                      for flow, flow_data in self.flows.items()
                      if len(flow_data) is not 0}

    def add_sample(self, flow, sample, absolute_timestamp):
        """
        Adds [flow] to the map of all recorded flows with an initial sample of [sample] and an start time of
        [absolute_timestamp].
        If [flow] is already mapped to some data stream - the new sample will be added to that stream instead
        """
        if flow in self.flows:
            self.flows[flow].add_sample(sample)
        else:
            self.flows[flow] = FlowData(sample, absolute_start_time=absolute_timestamp)

    def compose_new_batch(self, current_absolute_time):
        """
        :return: list of 2-tuples of (flow 5-tuple, block of last 60 secs), for all flows in the map that exist at least
        [BLOCK_LENGTH] seconds
        :param current_absolute_time: the current time for flows' start to be compared to
        """
        batch = []
        for flow, flow_data in self.flows.items():
            time_flow_lives = current_absolute_time - flow_data.absolute_start_time
            if time_flow_lives >= BLOCK_LENGTH:
                logging.debug(f'pushing flow {flow}, which exists for {time_flow_lives} seconds')
                batch.append((flow, flow_data.to_block()))
            else:
                logging.debug(f'skipping flow {flow}, which only exists for {time_flow_lives} seconds')
        logging.debug(f'number of RELEVANT blocks created during window slide: {len(batch)}')
        logging.debug(f'number of IRRELEVANT mapped during window slide: {len(self.flows) - len(batch)}')
        return batch


class LiveCaptureProvider:
    """
    Performs a live capture.
    Creates a new block after 60 seconds and every 15 seconds thereafter, and adds the flows to [queue].
    Each time a flow is created - all subscribers will be notified
    """

    def __init__(self):
        # only_summaries flag is important! contains packet size and arrival time (relative to capture start)
        # TODO: see if we need to add more capture filters
        capture_filter = 'ip and ' \
                         'port not 53 and ' \
                         'not broadcast and ' \
                         'not ip6 and ' \
                         'not igmp and ' \
                         'not icmp and ' \
                         'port not 123'

        self.capture = NonPromiscuousLiveCapture(capture_filter=capture_filter, only_summaries=True)
        self.capture.set_debug()
        self.queue = deque()
        self.absolute_start_time = None
        self.absolute_current_time = None
        self.relative_time = None
        self.sliding_window_advancements = 0
        self.flows_manager = FlowsManager()

    def packet_callback2(self, packet):
        logging.debug(packet)

    def packet_callback(self, packet):
        """
        Callback to be invoked on every new [packet] capture.
        The timestamp of [packet] advances the relative start time of this recording.
        If the new captured time is sitting at a new block interval (every [BLOCK_INTERVAL] seconds) - the sliding
        window will advance and a new batch of blocks will be created (starting at [BLOCK_LENGTH] seconds).

        The new [packet] capture's metadata is stored: 5-tuple for the flow and sample of [packet]'s arrival time and
        size are added to the flow.
        """
        if self.absolute_start_time is None:
            self.absolute_start_time = float(packet.sniff_timestamp)
        self.absolute_current_time = float(packet.sniff_timestamp)
        self.relative_time = self.absolute_current_time - self.absolute_start_time

        # logging.debug(f'absolute start time: {self.absolute_start_time}\nabsolute current time: '
        #               f'{self.absolute_current_time}\nrelative time: {self.relative_time}')

        # logging.debug(packet)

        self.advance_sliding_window_if_needed()

        transport = packet.transport_layer
        flow = (
            packet['ip'].src,
            packet[transport].srcport,
            packet['ip'].dst,
            packet[transport].dstport,
            transport
        )
        self.flows_manager.add_sample(flow, self._extract_packet_meta(packet), self.absolute_current_time)

    def advance_sliding_window_if_needed(self):
        """
        Advances the sliding window as many times as needed based on the last packet capture's timestamp.
        One window step = [BLOCK_INTERVAL] seconds.

        - In the vast majority of cases, since there are frequent packet captures, we will not advance the window at all
        - Other cases we will advance the window by one step
        - Very rarely (e.g. in case of no internet connection), we will advance the window by more than one step
        """
        while self.relative_time >= (self.sliding_window_advancements + 1) * BLOCK_INTERVAL:
            pre_update = datetime.datetime.now()
            self.sliding_window_advancements += 1
            if self.sliding_window_advancements >= BLOCK_LENGTH // BLOCK_INTERVAL:
                self.push_flows()
            self.flows_manager.advance_sliding_window()
            post_update = datetime.datetime.now()
            logging.debug(f'\ntime it took to push all flows and/or advance window: {post_update - pre_update}\n')

    def push_flows(self):
        """
        Pushes a new block for each flow in the map that has lived for at least [BLOCK_LENGTH] seconds
        """
        self.queue.append(self.flows_manager.compose_new_batch(self.absolute_current_time))
        # self.queue.clear()

    @staticmethod
    def _extract_packet_meta(packet: Packet) -> Tuple[float, int]:
        return float(packet.sniff_timestamp), int(packet['ip'].len)

    def start_capture(self):
        # self.capture.load_packets(timeout=10)
        # for packet in self.capture:
        self.capture.apply_on_packets(self.packet_callback2)

    def stop_capture(self):
        """
        TODO: change the overall structure of control over the live capture so that stopping actually does anything here
        """
        self.capture.close()


if __name__ == '__main__':
    live = LiveCaptureProvider()
    live.start_capture()
    # live.stop_capture()
