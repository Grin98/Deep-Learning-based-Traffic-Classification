import os
from os.path import splitext
from typing import Tuple

import numpy as np
import pyshark
import csv


def is_undesired_packet(packet):
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


def extract_packet_sampling(packet) -> Tuple[float, int]:
    if 'IP' in packet:
        ip = packet['IP']
        size = int(ip.len)
    else:
        ip = packet['IPv6']
        size = int(ip.len)
    return float(packet.sniff_timestamp), int(size)


if __name__ == '__main__':
    # c = pyshark.FileCapture("pcaps/netflix_1.pcapng")
    # print(c[0])
    flows = {}
    for file in os.listdir('pcaps'):
        filename, file_extension = splitext(file)
        if file_extension != '.pcap' and file_extension != '.pcapng':
            raise Exception(file_extension + ' file extension not supported')
        capture = pyshark.FileCapture('pcaps/' + filename + file_extension)

        for packet in capture:
            if is_undesired_packet(packet):
                continue

            transport = packet.transport_layer
            flow_five_tuple = (
                packet.ip.src,
                packet[transport].srcport,
                packet.ip.dst,
                packet[transport].dstport,
                transport
            )

            packet_sampling = extract_packet_sampling(packet)

            if flow_five_tuple in flows:
                # print("writing " + str(packet_sampling) + "to " + str(flow_five_tuple))
                flows[flow_five_tuple].append(packet_sampling)
            else:
                # print("writing " + str(packet_sampling) + "to new flow " + str(flow_five_tuple))
                flows[flow_five_tuple] = [packet_sampling]
        capture.close()

        out_file = 'parsed-flows/' + filename + '.csv'
        with open(out_file, 'w+', newline='') as f:
            writer = csv.writer(f, delimiter=',')

            max_five_tuple = max(flows, key=lambda x: len(flows[x]))
            max_flow_samples = flows[max_five_tuple]
            times, sizes = zip(*max_flow_samples)

            # normalize time to start from 0
            start = times[0]
            times = [t - start for t in times]

            row = ['app_place_holder'] + list(max_five_tuple) + [start] + [len(times)] + \
                  times + [' '] + list(sizes)
            writer.writerow(row)
        # app|src_ip|src_port|dst_ip|dst_port|transport_protocol|start_time|length|[timestamps]|[sizes]|
