from os import listdir
from os.path import isfile, join, splitext
from scapy.all import *
import scapy.layers
import scapy.layers.inet
import datetime


def printType(obj):
    print(type(obj))


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


if __name__ == '__main__':
    dataDirectoryPath = "dataset samples"
    metaDirectoryPath = "metaSamples"

    filesNames = [f for f in listdir(dataDirectoryPath) if isfile(join(dataDirectoryPath, f))]
    if not os.path.exists(metaDirectoryPath):
        os.mkdir(metaDirectoryPath)
    print(filesNames)

    for fileFullName in filesNames:
        print("parsing " + fileFullName)

        fileName, fileExtension = splitext(fileFullName)
        pcapFile = open(join(dataDirectoryPath, fileFullName), 'br')
        metadata = open(join(metaDirectoryPath, fileName + ".txt"), 'w+')

        packets = ''
        if fileExtension == '.pcap' or fileExtension == '.pcapng':
            packets = rdpcap(pcapFile)
        else:
            print("illegal extension:" + fileExtension)
            continue

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
            protocol = ''
            if 'TCP' in p:
                transport = p['TCP']
                protocol = 'tcp'
            if 'UDP' in p:
                transport = p['UDP']
                protocol = 'udp'
            if 'ICMP' in p:
                transport = p['ICMP']
                protocol = 'icmp'

            size = len(p)
            arrivalTime = p.time
            fiveTuple = [ip.src, transport.sport, ip.dst, transport.dport, protocol]
            metadata.write(str(fiveTuple) + " " + str(size) + " " + str(arrivalTime) + "\n")

        pcapFile.close()
        metadata.close()
