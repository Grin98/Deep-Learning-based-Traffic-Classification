import sys

sys.path.append("../")
sys.path.append("./")

from flowpic_dataset.loader import FlowPicDataLoader
from flowpic_dataset.processors import NoOverlapPreProcessor

if __name__ == '__main__':
    p = NoOverlapPreProcessor('data_tor')
    p.process_dataset('classes_tor')
    print('\n==========\n')

    p = NoOverlapPreProcessor('data_vpn')
    p.process_dataset('classes_vpn')
    print('\n==========\n')

    p = NoOverlapPreProcessor('data_reg')
    p.process_dataset('classes_reg')
    print('\n==========\n')

    FlowPicDataLoader('../data_reg', testing=False).load_dataset()
    FlowPicDataLoader('../data_tor', testing=False).load_dataset()
    FlowPicDataLoader('../data_vpn', testing=False).load_dataset()
