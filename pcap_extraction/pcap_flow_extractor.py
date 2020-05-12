import os
from os.path import splitext
import pyshark


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
    elif 'DNS' in packet:
        return True

    return False


def extract_packet_sampling(packet):
    if 'IP' in packet:
        ip = packet['IP']
        size = int(ip.len)
    else:
        ip = packet['IPv6']
        size = int(ip.len)
    return float(packet.sniff_timestamp), size


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
        f = open('parsed-flows/' + filename + '.data', 'w')
        max_five_tuple = max(flows, key=lambda x: len(set(flows[x])))
        max_flow_samples = flows[max_five_tuple]

        # src_ip|src_port|dst_ip|dst_port|transport_protocol|start_time|length|[timestamps]|[sizes]|

        '''
        print(max_five_tuple[0] + '|' +
                max_five_tuple[1] + '|' +
                max_five_tuple[2] + '|' +
                max_five_tuple[3] + '|' +
                max_five_tuple[4] + '|' +
                str(max_flow_samples[0][0]) + '|' +
                str(len(max_flow_samples)) + '|')
        '''

        f.write(max_five_tuple[0] + '|' +
                max_five_tuple[1] + '|' +
                max_five_tuple[2] + '|' +
                max_five_tuple[3] + '|' +
                max_five_tuple[4] + '|' +
                str(max_flow_samples[0][0]) + '|' +
                str(len(max_flow_samples)) + '|')
        for sample in max_flow_samples:
            # print(str(sample[0]) + ' ')
            f.write(str(sample[0]) + ' ')
        f.write('|')
        for sample in max_flow_samples:
            # print(str(sample[1]) + ' ')
            f.write(str(sample[1]) + ' ')
        f.write('|')
        f.close()
