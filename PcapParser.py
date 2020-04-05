from os.path import splitext
import pyshark
from pyshark import FileCapture
from pyshark.packet.packet import Packet
from pyshark.packet.packet_summary import PacketSummary
from pyshark.packet.layer import Layer
from pyshark.packet.fields import LayerFieldsContainer, LayerField

class PcapParser:
    def parse(self, input_file_path):
        _, file_extension = splitext(input_file_path)

        if file_extension != '.pcap' and file_extension != '.pcapng':
            print("illegal extension:" + file_extension)
            return None

        packets = pyshark.FileCapture(input_file_path)

        flows = {}
        for i, packet in enumerate(packets):
            print("packet number %d" % (i + 1))

            p: Packet = packet
            if 'IP' in p:
                ip: Layer = p['IP']
                size = int(ip.len)
            elif 'IPv6' in p:
                ip: Layer = p['IPv6']
                print(ip.field_names)
                size = int(ip.plen)
            else:
                print("No IP Layer")
                continue

            if p.transport_layer is None:
                print("No Transport Layer")
                continue

            highest_protocol = p.highest_layer
            transport_protocol = p.transport_layer
            transport: Layer = p[p.transport_layer]

            arrival_time = float(p.sniff_timestamp)
            src_ip = ip.src
            dst_ip = ip.dst
            src_port = transport.srcport
            dst_port = transport.dstport

            # TODO figure out what len to use
            five_tuple = (src_ip, src_port, dst_ip, dst_port, transport_protocol)
            if five_tuple in flows:
                flows[five_tuple].append((size, arrival_time))
            else:
                flows[five_tuple] = [(size, arrival_time)]

        packets.close()
        return flows
