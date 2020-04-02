from os import listdir
from os.path import isfile, join, splitext
from scapy.all import *
from PcapParser import PcapParser
from FlowPicBuilder import FlowPicBuilder


def print_type(obj):
    print(type(obj))


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
        metadata = open(join(metaDirectoryPath, fileName + ".txt"), 'w+')

        flows = PcapParser().parse(join(dataDirectoryPath, fileFullName))
        if flows is None:
            continue

        fpb = FlowPicBuilder()
        for key in flows:
            x = fpb.build_pic(flows[key])
            # TODO check option to save tensors as json and strings
            metadata.write(str(x))

        metadata.close()
