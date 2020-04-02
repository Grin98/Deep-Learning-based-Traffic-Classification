from os.path import splitext
from scapy.all import *


class PcapParser:
    def parse(self, input_file_path):
        _, file_extension = splitext(input_file_path)
        pcap_file = open(input_file_path, 'br')

        if file_extension == '.pcap' or file_extension == '.pcapng':
            packets = rdpcap(pcap_file)
        else:
            print("illegal extension:" + file_extension)
            return None

        flows = {}
        for i, p in enumerate(packets):
            print("packet number %d" % (i + 1))

            if 'IP' in p:
                ip = p['IP']
            elif 'IPv6' in p:
                ip = p['IPv6']
            else:
                if 'ARP' not in p:
                    print(p.show())
                continue

            transport = ''  # temp init
            protocol = ''  # temp init
            if 'TCP' in p:
                transport = p['TCP']
                protocol = 'tcp'
            elif 'UDP' in p:
                transport = p['UDP']
                protocol = 'udp'
            elif 'ICMP' in p:
                continue

            size = ip.len
            arrival_time = p.time
            five_tuple = (ip.src, transport.sport, ip.dst, transport.dport, protocol)
            if five_tuple in flows:
                flows[five_tuple].append((size, arrival_time))
            else:
                flows[five_tuple] = [(size, arrival_time)]

        return flows
