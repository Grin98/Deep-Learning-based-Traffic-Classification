from pyshark import LiveCapture
from collections import deque
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


class LiveCaptureProvider:
    """
    Performs a live capture.
    Creates a new flow after 60 seconds and every 15 seconds thereafter, and adds the flows to [queue].
    Each time a flow is created - all subscribers are notified

    :raises [exception] if no interfaces are found
            [exception] if no packets are found
    """

    def __init__(self):
        # only_summaries flag is extremely important! contains packet size and arrival time (relative to capture start)
        self.capture = NonPromiscuousLiveCapture(only_summaries=True)
        self.queue = deque()
        self.relative_time = 0.0

    def packet_callback(self, packet):
        self.relative_time = self.relative_time + 1
        print(self.relative_time)

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
    live.stop_capture()
