from os.path import isfile, join, splitext, isdir
from os import listdir
import os
import csv
from typing import List, Tuple, NamedTuple
import numpy as np


class Packet(NamedTuple):
    size: int
    time: float


class SubFlow:
    def __init__(self, start: float, labels: List[str], writer):

        self.labels = labels
        self.writer = writer
        self.start = start
        self.packets = []
        self.has_overlapped = False

    def add_packet(self, p: Packet):
        self.packets.append((p.size - 1, p.time - self.start))

    def write_to_file(self):
        if not self.packets:
            return

        sizes, times = zip(*self.packets)
        sizes = list(sizes)
        times = list(times)
        data = self.labels + [len(self.packets)] + times + sizes
        self.writer.writerow(data)


class PreProcessor:
    def __init__(self, dataset_dir, processed_dir,
                 flow_duration_in_seconds: int = 60,
                 overlap_in_seconds: int = 15,
                 packet_size_limit: int = 1500):
        self.inPath = dataset_dir
        self.outPath = processed_dir
        self.flow_duration = flow_duration_in_seconds
        self.overlap = overlap_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_dataset(self):
        self.__process_dirs__(self.inPath, self.outPath, [])

    def __process_dirs__(self, input_dir_path, output_dir_path, labels: List[str]):

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        dirs = [d for d in listdir(input_dir_path) if isdir(join(input_dir_path, d))]
        if not dirs:
            # there are only files
            for name in listdir(input_dir_path):
                self.__process_file__(join(input_dir_path, name), join(output_dir_path, name), labels)

        for name in dirs:
            labels.append(name)
            self.__process_dirs__(join(input_dir_path, name), join(output_dir_path, name), labels)
            labels.pop()

    def __process_file__(self, input_file_path: str, output_file_path: str, labels: List[str]):

        _, file_extension = splitext(input_file_path)
        if file_extension != '.csv':
            return

        print('processing ' + input_file_path)

        with open(output_file_path, 'w+', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')

            with open(input_file_path, newline='') as f_in:
                data = csv.reader(f_in, delimiter=',')
                for row in data:

                    app, flow = self.__transform_row_to_flow__(row)
                    labels.append(app)
                    sub_flows: List[SubFlow] = [SubFlow(flow[0].time, labels, writer)]
                    for p in flow:
                        if p.size > self.packet_size_limit:
                            continue

                        self.__add_packet_to_sub_flows__(sub_flows, p)

                    for f in sub_flows:
                        f.write_to_file()
                    labels.pop()


    @staticmethod
    def __transform_row_to_flow__(row: List[str]) -> Tuple[str, List[Packet]]:
        app = row[0]
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        return app, [Packet(size=s, time=t) for s, t in zip(sizes, times)]

    def __add_packet_to_sub_flows__(self, sub_flows: List[SubFlow], p: Packet):
        for f in sub_flows:
            time_passed = p.time - f.start
            if time_passed >= self.flow_duration:
                f.write_to_file()
                sub_flows.remove(f)

                if not f.has_overlapped:
                    self.__add_new_flow__(sub_flows, p.time, f.labels, f.writer)
                continue

            if time_passed >= (self.flow_duration - self.overlap) and not f.has_overlapped:
                self.__add_new_flow__(sub_flows, p.time, f.labels, f.writer)
                f.has_overlapped = True

            f.add_packet(p)

    @staticmethod
    def __add_new_flow__(sub_flows: List[SubFlow], start: float, labels: List[str], writer):
        new_flow = SubFlow(start, labels, writer)
        sub_flows.append(new_flow)

    # def __splitFlow__(self, flow: List[Tuple[int, float]]):
    #     start = flow[0][1]
    #     splitted_flows = []
    #     sub_flow = []
    #     for size, time in flow:
    #         # throw too large packets
    #         if size > self.packet_size_limit:
    #             continue
    #
    #         time_passed = time - start
    #         if time_passed >= self.flow_duration:
    #             start = time
    #             splitted_flows.insert(0, sub_flow.copy())
    #             sub_flow = []
    #
    #         # normalize time of the packets in the sub flow
    #         # and decrease size by 1 to fit the indexing of the tensor
    #         sub_flow.append((size - 1, time - start))
    #
    #     # if sub_flow isn't empty add it
    #     if sub_flow:
    #         splitted_flows.insert(0, sub_flow)
    #
    #     return splitted_flows
