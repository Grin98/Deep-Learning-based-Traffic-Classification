import dpkt
import datetime
import socket
from dpkt.compat import compat_ord
from os import listdir
from os.path import isfile, join, splitext


def printType(obj):
    print(type(obj))


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


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
    filesNames = [f for f in listdir(dataDirectoryPath) if isfile(join(dataDirectoryPath, f))]

    print(filesNames)
    for fileName in filesNames:
        print("parsing " + fileName)

        f = open(join(dataDirectoryPath, fileName), 'br')
        _, fileExtension = splitext(fileName)
        reader = ''
        if fileExtension == '.pcap':
            reader = dpkt.pcap.Reader(f)
        elif fileExtension == '.pcapng':
            reader = dpkt.pcapng.Reader(f)
        else:
            print("illegal extension:" + fileExtension)
            continue

        for i, (time, buf) in enumerate(reader):
            print("packet number %d" % (i + 1))
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            transport = ip.data

            print("packet arrival time %s" % str(datetime.datetime.utcfromtimestamp(time)))
            print("packet size %d" % len(buf))
            if type(ip) is not dpkt.arp.ARP:
                print("ips: %s -> %s" % (inet_to_str(ip.src), inet_to_str(ip.dst)))
            else:
                print("Packet protocol is ARP")
            print("macs: %s -> %s" % (mac_addr(eth.src), mac_addr(eth.dst)))
            print("")
        f.close()
        break
