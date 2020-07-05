import subprocess as sp
from pyshark import LiveCapture, capture
from collections import deque
from misc.data_classes import Block
from misc.constants import PROFILE, BLOCK_DURATION, BLOCK_INTERVAL
import threading
import itertools
import datetime
import time
import logging

# TODO: CHANGE LOGGING LEVEL FROM [logging.DEBUG] TO [logging.INFO] TO STOP DEBUG MESSAGES!!!
# logging.basicConfig(filename='example_live_capture.txt', format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True


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


def _anti_clockwise_crange(start, modulo):
    """
    A generator for an anti-clockwise cyclic range
    :param start: number of steps anti-clockwise from 0 to start at
    :param modulo: the number of steps in the cycle
    """
    for i in range(modulo):
        yield (start + i) % modulo


class FlowData:
    """
    Data class that represents an ongoing flow's data stream, divided into 4 windows.
    The class handles sliding window shifts very quickly by maintaining a cyclic ring buffer.
    """

    def __init__(self, sample):
        """
        :param sample: a 2-tuple of (arrival_time, packet_size)
        :param absolute_start_time: timestamp of the first packet in the flow
        """
        # these are windows with the samples from the flow during the last: [0,15], [15,30], [30,45] and [45,60]
        # seconds. self.head refers to the "most recent" window, and every *previous* index modulo 4 will be the next
        self.window_samples = [[] for _ in range(BLOCK_DURATION // BLOCK_INTERVAL)]
        self.head = 0
        self.window_samples[self.head].append(sample)
        self.absolute_start_time = sample[0]
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
        self.head = (self.head + 1) % len(self.window_samples)
        self.alive_intervals += 1
        return self

    def __len__(self):
        return sum(len(window) for window in self.window_samples)

    def to_block(self, num_slides):
        """
        :return: a block of the last 60 secs
        """

        # print(f"for flow that started at {self.absolute_start_time} with head: "
        #       f"{(self.head + 1) % len(self.window_samples)} for block that starts at: "
        #       f"{num_slides * BLOCK_INTERVAL - BLOCK_DURATION}")
        # for i in _anti_clockwise_crange((self.head + 1) % len(self.window_samples),
        #                                 len(self.window_samples)):
        #     if len(self.window_samples[i]) != 0:
        #         print(f"win{i} 1st timestamp: {self.window_samples[i][0][0]} ; ", end='')
        # if len(self.window_samples[self.head]) > 0:
        #     print(f"last timestamp: {self.window_samples[self.head][-1][0]}", end='')
        # print()

        block = Block.create_from_stream(start_time=
                                         num_slides * BLOCK_INTERVAL - BLOCK_DURATION,
                                         data=list(itertools.chain.from_iterable(
                                             self.window_samples[i] for i in _anti_clockwise_crange(
                                                 (self.head + 1) % len(self.window_samples),
                                                 len(self.window_samples))))
                                         )
        self.window_samples[(self.head + 1) % len(self.window_samples)].clear()
        return block


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
                      if len(flow_data) > 0}

    def add_sample(self, flow, sample):
        """
        Adds [flow] to the map of all recorded flows with an initial sample of [sample] and an start time of
        [absolute_timestamp].
        If [flow] is already mapped to some data stream - the new sample will be added to that stream instead
        """
        if flow in self.flows:
            self.flows[flow].add_sample(sample)
        else:
            self.flows[flow] = FlowData(sample)

    def compose_new_batch(self, current_absolute_time, num_slides):
        """
        :return: list of 2-tuples of (flow 5-tuple, block of last 60 secs), for all flows in the map that exist at least
        [BLOCK_LENGTH] seconds
        :param current_absolute_time: the current time for flows' start to be compared to
        """
        batch = []
        for flow, flow_data in self.flows.items():
            time_flow_lives = current_absolute_time - flow_data.absolute_start_time
            if time_flow_lives > BLOCK_DURATION - BLOCK_INTERVAL and len(flow_data) > 0:
                logging.debug(f'pushing flow {flow}, which exists for {time_flow_lives} seconds')
                batch.append((flow, flow_data.to_block(num_slides)))
            else:
                logging.debug(f'skipping flow {flow}, which only exists for {time_flow_lives} seconds')
        logging.debug(f'number of RELEVANT blocks created during window slide: {len(batch)}')
        logging.debug(f'number of IRRELEVANT flows mapped during window slide: {len(self.flows) - len(batch)}')
        return batch


class LiveCaptureProvider:
    """
    Performs a live capture.
    Creates a new block after 60 seconds and every 15 seconds thereafter, and adds the flows to [queue].
    Each time a flow is created - all subscribers will be notified
    """

    def __init__(self, interfaces, save_to_file=False):
        # TODO: see if we can find more capture filters to add
        capture_filter = 'ip and ' \
                         'port not 53 and ' \
                         'not broadcast and ' \
                         'not ip6 and ' \
                         'not igmp and ' \
                         'not icmp and ' \
                         'port not 123'

        # RECOMMEND: KEEP THIS AT FALSE!
        if save_to_file:
            output_file = str(time.strftime("%Y-%m-%d_%H-%M-%S.pcapng"))
            self.capture = NonPromiscuousLiveCapture(interface=interfaces,
                                                     only_summaries=True,
                                                     capture_filter=capture_filter,
                                                     output_file=output_file)
        else:
            self.capture = NonPromiscuousLiveCapture(interface=interfaces,
                                                     only_summaries=True,
                                                     capture_filter=capture_filter)

        self.capture.set_debug()
        self.queue = deque()
        self.absolute_start_time = None
        self.absolute_current_time = None
        self.relative_time = None
        self.sliding_window_advancements = 0
        self.flows_manager = FlowsManager()
        self.window_lock = threading.Lock()
        self.terminate = False

    def packet_callback(self, packet):
        """
        Callback to be invoked on every new [packet] capture.
        The timestamp of [packet] advances the relative start time of this recording.
        If the new captured time is sitting at a new block interval (every [BLOCK_INTERVAL] seconds) - the sliding
        window will advance and a new batch of blocks will be created (starting at [BLOCK_LENGTH] seconds).

        The new [packet] capture's metadata is stored: 5-tuple for the flow and sample of [packet]'s arrival time and
        size are added to the flow.
        """

        packet_sample = (float(packet._fields['Time']), int(packet._fields['Length']))

        if self.absolute_start_time is None:
            self.absolute_start_time = packet_sample[0]
        self.absolute_current_time = packet_sample[0]
        self.relative_time = self.absolute_current_time - self.absolute_start_time

        with self.window_lock:
            while self.relative_time >= (self.sliding_window_advancements + 1) * BLOCK_INTERVAL:
                self.advance_sliding_window()

        flow = (
            packet._fields['Source'],
            packet._fields['SrcPort'],
            packet._fields['Destination'],
            packet._fields['DstPort'],
            packet._fields['Protocol']
        )

        self.flows_manager.add_sample(flow, packet_sample)

    def advance_sliding_window(self):
        """
        Advances the sliding window once.
        One window step = [BLOCK_INTERVAL] seconds.

        - For the majority of packets, since there are frequent packet captures, we will not advance the window at all
        - Other cases we will advance the window by one step
        - Very rarely (e.g. in case of internet disconnection), we will advance the window by more than one step
        """
        pre_update = datetime.datetime.now()
        self.sliding_window_advancements += 1
        if self.relative_time >= BLOCK_DURATION:
            self.push_flows(self.sliding_window_advancements)
        self.flows_manager.advance_sliding_window()
        post_update = datetime.datetime.now()
        logging.debug(f'\ntime it took to push all flows and/or advance window: {post_update - pre_update}\n')

    def push_flows(self, num_slides):
        """
        Pushes a new block for each flow in the map that has lived for at least [BLOCK_LENGTH] seconds
        """
        self.queue.append(self.flows_manager.compose_new_batch(self.absolute_current_time, num_slides))
        # self.queue.clear()

    @staticmethod
    def get_net_interfaces():
        cmd_line = ["dumpcap", "-D"]
        output = sp.check_output(cmd_line).decode('utf-8')
        return [line[line.find("(") + 1:line.find(")")] for line in output.splitlines()]

    def start_capture(self):
        # self.capture.load_packets(timeout=10)
        # for packet in self.capture:

        try:
            for packet in self.capture.sniff_continuously():
                if self.terminate:
                    break
                self.packet_callback(packet)
        except capture.capture.TSharkCrashException as error:
            print(error)

        # self.capture.apply_on_packets(self.packet_callback)

    def stop_capture(self):
        """
        TODO: make sure that this actually terminates the packet capture processes
        """
        self.terminate = True


if __name__ == '__main__':
    live = LiveCaptureProvider(LiveCaptureProvider.get_net_interfaces())
    live.start_capture()
