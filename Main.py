from os import listdir
from os.path import isfile, join, splitext
from scapy.all import *
from PcapParser import PcapParser
from FlowPicBuilder import FlowPicBuilder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
        with open(join(metaDirectoryPath, fileName + ".txt"), 'w+') as metadata:

            flows = PcapParser().parse(join(dataDirectoryPath, fileFullName))
            if flows is None:
                continue

            fpb = FlowPicBuilder()
            for key in flows:
                pics = fpb.build_pic(flows[key])

                # for pic in pics:
                #     x = pic[1]
                #     if pic[0] == FlowPicBuilder.PicFormat.Sparse:
                #         x = x.to_dense()
                #     x = x.numpy()
                #     x[x >= 1] = 1
                #     if np.count_nonzero(x) >= 500:
                #         print(x)
                #         plt.imshow(x, cmap='binary')
                #         plt.show()
                #         exit()


                # TODO check option to save tensors as json and not strings
                metadata.write(str(key))
                metadata.write(str(pics))
