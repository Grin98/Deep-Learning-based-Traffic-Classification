import random
from os.path import isfile, join, splitext, isdir
from os import listdir
import os
import csv
from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path


class BaseProcessor(ABC):
    def __init__(self,
                 block_duration_in_seconds: int,
                 block_delta_in_seconds: int,
                 packet_size_limit: int):

        self.block_duration = block_duration_in_seconds
        self.block_delta = block_delta_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_dataset(self, dataset_dir: str):
        self._process_dirs(Path(dataset_dir))
        print('finished processing')

    def _process_dirs(self, input_dir_path: Path):
        dirs = [d for d in input_dir_path.glob('*') if d.is_dir()]
        if not dirs:
            self._process_dir_files(input_dir_path)

        else:
            for d in dirs:
                self._process_dirs(d)

    @abstractmethod
    def _process_dir_files(self, input_dir_path: Path):
        pass

    def __process_flows__(self, flows):
        blocks = []
        for app, sizes, times in flows:
            blocks += self.__split_flow_to_blocks__(times, sizes)

        return blocks

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
        sizes = sizes[mask] - 1

        return app, sizes, times

    def __split_flow_to_blocks__(self, times, sizes):
        num_blocks = int(times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1
        blocks = []
        for b in range(num_blocks):
            start = b * self.block_delta
            end = b * self.block_delta + self.block_duration

            mask = ((times >= start) & (times <= end))
            if np.count_nonzero(mask) == 0:
                continue

            block_times = times[mask]
            block_sizes = sizes[mask]

            # normalize times to start from 0
            block_times = block_times - b * self.block_delta

            block = [len(block_sizes)] + block_times.tolist() + block_sizes.tolist()
            blocks.append(block)
        return blocks

    @staticmethod
    def __create_dir__(dir: Path):
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

    def _create_output_dir(self, out_root_dir: Path, input_dir_path: Path):
        sub_path = Path(*input_dir_path.parts[1:])
        out_path_dir = out_root_dir / sub_path
        self.__create_dir__(out_path_dir)
        return out_path_dir


class SplitProcessor(BaseProcessor):
    """
    statically splits dataset of flows to Train and Test sets before splitting each flow to blocks
    and in addition there is an overlap between consecutive blocks depending on the value of [block_delta_in_seconds]
    """

    def __init__(self, test_dir, train_dir, block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15, packet_size_limit: int = 1500):
        super().__init__(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        self.testPath = Path(test_dir)
        self.trainPath = Path(train_dir)

    def _process_dir_files(self, input_dir_path: Path):
        out_file = 'data.csv'
        test_file_path = self._create_output_dir(self.testPath, input_dir_path) / out_file
        train_file_path = self._create_output_dir(self.trainPath, input_dir_path) / out_file

        with test_file_path.open('w+', newline='') as test_out:
            test_writer = csv.writer(test_out, delimiter=',')

            with train_file_path.open('w+', newline='') as train_out:
                train_writer = csv.writer(train_out, delimiter=',')

                test_blocks = []
                train_blocks = []
                for file in input_dir_path.glob('*'):
                    file_test_blocks, file_train_blocks = self._process_file(file)
                    test_blocks += file_test_blocks
                    train_blocks += file_train_blocks

                self.__sample_and_write_blocks__(test_blocks, test_writer, approximate_amount=300)
                self.__sample_and_write_blocks__(train_blocks, train_writer, approximate_amount=3000)

    def _process_file(self, input_file_path: Path):

        file_extension = input_file_path.suffix
        if file_extension != '.csv':
            return [], []

        print('processing ' + str(input_file_path))

        with input_file_path.open(newline='') as f_in:
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

            test_blocks = self.__process_flows__(test_flows)
            train_blocks = self.__process_flows__(train_flows)

            return test_blocks, train_blocks

    @staticmethod
    def __sample_and_write_blocks__(blocks, writer, approximate_amount: int):
        if len(blocks) <= approximate_amount:
            condition = lambda: True
        else:
            prob = approximate_amount / len(blocks)
            condition = lambda: random.random() < prob

        for b in blocks:
            if condition():
                writer.writerow(b)


class NoOverlapProcessor(BaseProcessor):
    def __init__(self,
                 out_root_dir_path: str,
                 block_duration_in_seconds: int = 60,
                 packet_size_limit: int = 1500):

        super().__init__(block_duration_in_seconds, block_duration_in_seconds, packet_size_limit)
        self.out_root_dir = Path(out_root_dir_path)

    def _process_dir_files(self, input_dir_path: Path):
        out_file = 'data.csv'
        out_file_path = self._create_output_dir(self.out_root_dir, input_dir_path) / out_file

        with out_file_path.open('w+', newline='') as out_f:
            writer = csv.writer(out_f, delimiter=',')

            for file in input_dir_path.glob('*'):
                self._process_file(file, writer)

    def _process_file(self, input_file_path: Path, writer):
        file_extension = input_file_path.suffix
        if file_extension != '.csv':
            return

        print('processing ' + str(input_file_path))
        with input_file_path.open(newline='') as f_in:
            data = csv.reader(f_in, delimiter=',')

            for row in data:
                app, sizes, times = self.__transform_row_to_flow__(row)
                blocks = self.__split_flow_to_blocks__(times, sizes)
                for b in blocks:
                    writer.writerow(b)

