from enum import Enum
from math import floor

import torch
from typing import List, Tuple, Sequence


class FlowPicBuilder:

    def __init__(self, stream_duration_in_seconds: int = 60, pic_width: int = 1500, pic_height: int = 1500,
                 hist_cap: int = 255):
        self.stream_duration = stream_duration_in_seconds
        self.pic_width = pic_width
        self.pic_height = pic_height
        self.hist_cap = hist_cap
        self.hist = torch.zeros((pic_width, pic_height), dtype=torch.int16)

    def build_pic(self, stream: Sequence[Tuple[float, int]]):
        # scaling self.flow_duration in seconds to pic's pixel width
        x_axis_to_second_ratio = self.pic_width * 1.0 / self.stream_duration

        # reset hist
        self.hist = torch.zeros(self.pic_width, self.pic_height)

        for packet in stream:
            # packet is (time, size)
            x_position = int(floor(float(packet[0]) * x_axis_to_second_ratio))
            y_position = packet[1]
            if x_position >= 1500 or y_position >= 1500:
                print('x,y = ', x_position, y_position)
            self.hist[x_position][y_position] += 1

        pic = self.hist.clone()  # self.hist.clamp_max(self.hist_cap) / self.hist_cap
        return pic

