from enum import Enum
from math import floor

import torch
from typing import List, Tuple


class FlowPicBuilder:
    Flow = List[Tuple[int, float]]

    def __init__(self, flow_duration_in_seconds: int = 60, pic_width: int = 1500, pic_height: int = 1500):
        self.flow_duration = flow_duration_in_seconds
        self.pic_width = pic_width
        self.pic_height = pic_height
        self.hist = torch.zeros((pic_width, pic_height), dtype=torch.int16)

    def build_pic(self, flow: Flow):
        x_granularity = self.flow_duration * 1.0 / self.pic_width  # difference in seconds between hist indices
        flows = self.__splitFlow__(flow)

        flow_pics = []
        for f in flows:
            self.hist = torch.zeros(self.pic_width, self.pic_height)
            counter = 0
            for packet in f:
                x_position = int(round(float(packet[1]) * x_granularity))
                y_position = packet[0]
                if self.hist[x_position][y_position] == 0:
                    counter += 1
                self.hist[x_position][y_position] += 1

            # TODO decide if it's even worth it to save pics in different formats
            if counter < (1.0/3) * self.pic_width * self.pic_height:  # less than third of the hist was changed
                pic = self.hist.to_sparse()  # changing to sparse to save on memory
                pic_format = FlowPicBuilder.PicFormat.Sparse
            else:
                pic = self.hist.clone()
                pic_format = FlowPicBuilder.PicFormat.Dense

            flow_pics.insert(0, (pic_format, pic))

        return flow_pics

    def __splitFlow__(self, flow: Flow):
        start = flow[0][1]
        splitted_flows = []
        sub_flow = []
        for size, time in flow:
            # throw too large packets
            if size > self.pic_height:
                continue

            time_passed = time - start
            if time_passed >= self.flow_duration:
                start = time
                splitted_flows.insert(0, sub_flow.copy())
                sub_flow = []

            # normalize time of the packets in the sub flow
            # and decrease size by 1 to fit the indexing of the tensor
            sub_flow.append((size - 1, time - start))

        # if sub_flow isn't empty add it
        if sub_flow:
            splitted_flows.insert(0, sub_flow)

        return splitted_flows

    class PicFormat(Enum):
        Sparse = 1
        Dense = 2


