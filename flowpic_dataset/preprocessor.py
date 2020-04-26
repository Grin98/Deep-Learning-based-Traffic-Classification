from os.path import isfile, join, splitext, isdir
from os import listdir
import os
import csv
from typing import List, Tuple
import numpy as np

class PreProcessor:
    def __init__(self, dataset_dir, processed_dir,
                 block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500):
        self.inPath = dataset_dir
        self.outPath = processed_dir
        self.block_duration = block_duration_in_seconds
        self.block_delta = block_delta_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_dataset(self):
        self.__process_dirs__(self.inPath, self.outPath)
        print('finished processing')

    def __process_dirs__(self, input_dir_path, output_dir_path):

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        dirs = [d for d in listdir(input_dir_path) if isdir(join(input_dir_path, d))]
        if not dirs:
            # there are only files
            for name in listdir(input_dir_path):
                self.__process_file__(join(input_dir_path, name), join(output_dir_path, name))

        for name in dirs:
            self.__process_dirs__(join(input_dir_path, name), join(output_dir_path, name))

    def __process_file__(self, input_file_path: str, output_file_path: str):

        _, file_extension = splitext(input_file_path)
        if file_extension != '.csv':
            return

        print('processing ' + input_file_path)

        with open(output_file_path, 'w+', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')

            with open(input_file_path, newline='') as f_in:
                data = csv.reader(f_in, delimiter=',')
                for row in data:

                    app, sizes, times = self.__transform_row_to_flow__(row)
                    num_blocks = int(times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1
                    for b in range(num_blocks):
                        start = b * self.block_delta
                        end = b * self.block_delta + self.block_duration

                        mask = ((times >= start) & (times <= end))
                        block_times = times[mask]
                        block_sizes = sizes[mask]

                        # normalize times to start from 0
                        block_times = block_times - b*self.block_delta

                        block = [len(block_sizes)] + block_times.tolist() + block_sizes.tolist()
                        writer.writerow(block)

    def __transform_row_to_flow__(self, row: List[str]) -> Tuple:
        app = row[0]
        num_packets = int(row[7])
        off_set = 8  # meta data occupies first 8 indices
        times = row[off_set:(num_packets + off_set)]
        sizes = row[(num_packets + off_set + 1):]  # +1 because there is an empty cell between times and sizes

        # casting from string
        times = np.array(times, dtype=np.float)
        sizes = np.array(sizes, dtype=np.int)

        mask = sizes <= self.packet_size_limit
        times = times[mask]
        sizes = sizes[mask]

        return app, sizes, times
