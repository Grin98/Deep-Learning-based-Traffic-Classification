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
        # scaling self.flow_duration in seconds to pic's pixel width
        x_axis_to_second_ratio = self.pic_width * 1.0 / self.flow_duration

        # reset hist
        self.hist = torch.zeros(self.pic_width, self.pic_height)

        for packet in flow:
            # packet is (size, time)
            x_position = int(floor(float(packet[1]) * x_axis_to_second_ratio))
            y_position = packet[0]
            self.hist[x_position][y_position] += 1

        return self.hist.clone()

