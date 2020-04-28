import random
from os.path import isfile, join, splitext, isdir
from os import listdir
import os
import csv
from typing import List, Tuple
import numpy as np


class PreProcessor:
    def __init__(self, dataset_dir, test_dir, train_dir,
                 block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500):
        self.inPath = dataset_dir
        self.testPath = test_dir
        self.trainPath = train_dir
        self.block_duration = block_duration_in_seconds
        self.block_delta = block_delta_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_dataset(self):
        self.__process_dirs__(self.inPath, self.testPath, self.trainPath)
        print('finished processing')

    def __process_dirs__(self, input_dir_path, test_dir_path, train_dir_path):

        self.__create_dir__(test_dir_path)
        self.__create_dir__(train_dir_path)

        dirs = [d for d in listdir(input_dir_path) if isdir(join(input_dir_path, d))]
        if not dirs:
            # there are only files
            for name in listdir(input_dir_path):
                self.__process_file__(join(input_dir_path, name),
                                      join(test_dir_path, name),
                                      join(train_dir_path, name))

        for name in dirs:
            self.__process_dirs__(join(input_dir_path, name),
                                  join(test_dir_path, name),
                                  join(train_dir_path, name))

    def __process_file__(self, input_file_path: str, test_file_path: str, train_file_path: str):

        _, file_extension = splitext(input_file_path)
        if file_extension != '.csv':
            return

        print('processing ' + input_file_path)

        with open(test_file_path, 'w+', newline='') as test_out:
            test_writer = csv.writer(test_out, delimiter=',')

            with open(train_file_path, 'w+', newline='') as train_out:
                train_writer = csv.writer(train_out, delimiter=',')

                with open(input_file_path, newline='') as f_in:
                    data = csv.reader(f_in, delimiter=',')

                    flows = [self.__transform_row_to_flow__(row) for row in data]
                    num_flows = len(flows)

                    # create ndarray of tuples
                    train_flows = np.empty(num_flows, dtype=object)
                    train_flows[:] = flows

                    # split
                    test_indices = random.sample(range(num_flows), max(1, int(num_flows * 0.1)))
                    test_flows = train_flows[test_indices]
                    train_flows = np.delete(train_flows, test_indices)

                    for app, sizes, times in train_flows:
                        self.__write_flow_as_blocks__(times, sizes, train_writer)

                    for app, sizes, times in test_flows:
                        self.__write_flow_as_blocks__(times, sizes, test_writer)




    def __transform_row_to_flow__(self, row: List[str]) -> Tuple:
        app = row[0]
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        mask = sizes < self.packet_size_limit
        times = times[mask]
        sizes = sizes[mask]

        return app, sizes, times

    #
    # def __split_flow__(self, times, sizes, test_percent: float = 0.1):
    #     end = times[-1]
    #     start = times[-1] * (1 - test_percent)
    #     test_mask = (times >= start) & (times <= end)
    #     train_mask = times < start
    #
    #     test_times = times[test_mask]
    #     test_sizes = sizes[test_mask]
    #     train_times = times[train_mask]
    #     train_sizes = sizes[train_mask]
    #
    #     test_times = test_times - test_times[0]
    #
    #     testb = len(range(int(test_times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1))
    #     trainb = len(range(int(train_times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1))
    #     totalb = len(range(int(times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1))
    #
    #     if totalb != 0:
    #         print(testb, trainb, totalb)
    #         print(testb/totalb, trainb/totalb)
    #         print('============')
    #
    #     return test_times, test_sizes, train_times, train_sizes

    def __write_flow_as_blocks__(self, times, sizes, writer):
        num_blocks = int(times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1
        for b in range(num_blocks):
            start = b * self.block_delta
            end = b * self.block_delta + self.block_duration

            mask = ((times >= start) & (times <= end))
            block_times = times[mask]
            block_sizes = sizes[mask]

            # normalize times to start from 0
            block_times = block_times - b * self.block_delta

            block = [len(block_sizes)] + block_times.tolist() + block_sizes.tolist()
            writer.writerow(block)


    def __create_dir__(self, dir: str):
        if not os.path.exists(dir):
            os.mkdir(dir)
