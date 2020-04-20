from os.path import isfile, join, splitext, isdir
from os import listdir
import os
import csv
from typing import List, Tuple
import numpy as np

Flow = List[Tuple[int, float]]


class PreProcessor:
    def __init__(self, dataset_dir, processed_dir,
                 flow_duration_in_seconds: int = 60,
                 packet_size_limit: int = 1500):
        self.inPath = dataset_dir
        self.outPath = processed_dir
        self.flow_duration = flow_duration_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_dataset(self):
        self.__process_dirs__(self.inPath, self.outPath)

    def __process_dirs__(self, input_dir_path, output_dir_path):

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        dirs = [d for d in listdir(input_dir_path) if isdir(join(input_dir_path, d))]
        if not dirs:
            for name in listdir(input_dir_path):
                self.__process_file__(join(input_dir_path, name), join(output_dir_path, name))

        for name in dirs:
            self.__process_dirs__(join(input_dir_path, name), join(output_dir_path, name))

    def __process_dir_files__(self, input_dir_path, output_dir_path):

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        files_names = [f for f in listdir(input_dir_path) if isfile(join(input_dir_path, f))]
        for name in files_names:
            self.__process_file__(join(input_dir_path, name), join(output_dir_path, name))

    def __process_file__(self, input_file_path, output_file_path):

        _, file_extension = splitext(input_file_path)
        if file_extension != '.csv':
            return

        with open(output_file_path, 'w+', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')

            with open(input_file_path, newline='') as f_in:

                data = csv.reader(f_in, delimiter=',')
                for row in data:

                    flow: Flow = self.__transform_row_to_flow__(row)
                    splitted_flows = self.__splitFlow__(flow)

                    for flow in splitted_flows:
                        sizes, times = zip(*flow)
                        sizes = list(sizes)
                        times = list(times)
                        writer.writerow([len(flow)] + times + sizes)

    def __transform_row_to_flow__(self, raw: List[str]):
        num_packets = int(raw[7])
        off_set = 8  # meta data occupies first 8 indices
        times = raw[off_set:(num_packets + off_set)]
        sizes = raw[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        return list(zip(sizes, times))

    def __splitFlow__(self, flow: Flow):
        start = flow[0][1]
        splitted_flows = []
        sub_flow = []
        for size, time in flow:
            # throw too large packets
            if size > self.packet_size_limit:
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
