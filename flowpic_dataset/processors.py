import random
import csv
from typing import Sequence, NamedTuple
from abc import ABC, abstractmethod

from misc.data_classes import Flow, Block
from misc.utils import *


class BasicProcessor:
    """
    provides the basic processing functions of flows
    """
    def __init__(self,
                 block_duration_in_seconds: int,
                 block_delta_in_seconds: int,
                 packet_size_limit: int
                 ):
        self.block_duration = block_duration_in_seconds
        self.block_delta = block_delta_in_seconds
        self.packet_size_limit = packet_size_limit

    def process_file_to_flows(self, input_file_path: Path):
        file_extension = input_file_path.suffix
        if file_extension != '.csv':
            return []

        with input_file_path.open(newline='') as f_in:
            data = csv.reader(f_in, delimiter=',')
            return [Flow.create(row, self.packet_size_limit) for row in data]

    def split_multiple_flows_to_blocks(self, flows: Sequence[Flow]) -> Sequence[Block]:
        blocks = []
        for f in flows:
            blocks += self.split_flow_to_blocks(f)

        return blocks

    def split_flow_to_blocks(self, flow: Flow) -> Sequence[Block]:
        times = flow.times
        sizes = flow.sizes
        num_blocks = max(int(times[-1] / self.block_delta - self.block_duration / self.block_delta) + 1, 1)

        blocks = []
        for b in range(num_blocks):
            start = b * self.block_delta
            end = b * self.block_delta + self.block_duration

            mask = ((times >= start) & (times < end))
            if np.count_nonzero(mask) == 0:
                continue

            block_times = times[mask]
            block_sizes = sizes[mask]
            num_packets = len(block_times)

            # normalize times to start from 0
            block_start_time = b * self.block_delta
            block_times = block_times - block_start_time

            block = Block(block_start_time, num_packets, list(zip(list(block_times), list(block_sizes))))
            blocks.append(block)

        return blocks

    @staticmethod
    def _write_blocks(blocks: Sequence[Block], writer, tag=None):
        if tag is None:
            tag = ''

        print('%s saving %d blocks' % (tag, len(blocks)))
        for b in blocks:
            writer.writerow(b.convert_to_row())


class DirectoriesProcessor(BasicProcessor, ABC):
    def process_dataset(self, dataset_dir: str):
        self._process_dirs(Path(dataset_dir))
        print('finished processing')

    def _process_dirs(self, input_dir_path: Path):
        dirs = get_dir_directories(input_dir_path)
        if not dirs:
            self._process_dir_files(input_dir_path)

        else:
            for d in dirs:
                self._process_dirs(d)

    @abstractmethod
    def _process_dir_files(self, input_dir_path: Path):
        pass


class SplitPreProcessor(DirectoriesProcessor):
    """
    statically splits dataset of flows to Train and Test sets before splitting each flow to blocks
    and in addition there is an overlap between consecutive blocks depending on the value of [block_delta_in_seconds]
    """

    def __init__(self, out_root_dir,
                 test_percent: float = 0.1,
                 train_size_cap: int = 2700,
                 test_size_cap: int = 300,
                 block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15, packet_size_limit: int = 1500):
        super().__init__(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        self.out_root_dir = Path(out_root_dir)
        self.train_path = out_root_dir / Path('train')
        self.test_path = out_root_dir / Path('test')
        self.test_percent = test_percent
        self.train_size_cap = train_size_cap
        self.test_size_cap = test_size_cap

    def _process_dir_files(self, input_dir_path: Path):
        out_file = 'data.csv'
        test_file_path = create_output_dir(self.test_path, input_dir_path) / out_file
        train_file_path = create_output_dir(self.train_path, input_dir_path) / out_file

        print('processing %s' % input_dir_path)
        with test_file_path.open('w+', newline='') as test_out:
            test_writer = csv.writer(test_out, delimiter=',')

            with train_file_path.open('w+', newline='') as train_out:
                train_writer = csv.writer(train_out, delimiter=',')

                flows = []
                for file in get_dir_csvs(input_dir_path):
                    flows += self.process_file_to_flows(file)

                train_flows, test_flows = self._split_train_test(flows, self.test_percent)
                train_blocks = self.split_multiple_flows_to_blocks(train_flows)
                test_blocks = self.split_multiple_flows_to_blocks(test_flows)

                train_blocks = self._sample_blocks(train_blocks, target_amount=self.train_size_cap)
                test_blocks = self._sample_blocks(test_blocks, target_amount=self.test_size_cap)

                self._write_blocks(train_blocks, train_writer, tag='train')
                self._write_blocks(test_blocks, test_writer, tag='test')

    @staticmethod
    def _split_train_test(flows, test_percent):
        num_flows = len(flows)
        train_flows = create_array_of_objects(flows)

        # split
        test_indices = random.sample(range(num_flows), max(1, int(num_flows * test_percent)))
        test_flows = train_flows[test_indices]
        train_flows = np.delete(train_flows, test_indices)

        return train_flows, test_flows

    @staticmethod
    def _sample_blocks(blocks: Sequence[Block], target_amount: int):
        current_amount = len(blocks)
        if current_amount > target_amount:
            indices = random.sample(range(current_amount), target_amount)
            blocks = create_array_of_objects(blocks)
            blocks = blocks[indices]

        return blocks


class QuickFlowFileProcessor:
    """
    splits file flows to raw blocks but doesn't write them out to a file,
    instead it returns them.
    """

    def __init__(self, block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500):

        self.p: BasicProcessor = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)

    def transform_file_to_blocks(self, file: Path) -> Sequence[Block]:
        flows = self.p.process_file_to_flows(file)
        return self.p.split_multiple_flows_to_blocks(flows)


class QuickPcapFileProcessor:
    def __init__(self, block_duration_in_seconds: int = 60,
                 block_delta_in_seconds: int = 15,
                 packet_size_limit: int = 1500):
        self.p: BasicProcessor = BasicProcessor(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
        # TODO add PcapParser (composition)

    def transform_pcap_to_groups_of_block_rows(self, file: Path):
        flows: Sequence[Flow]  # TODO add call for PcapParser to parse file and return flows
        return [self.p.split_flow_to_blocks(flow) for flow in flows]

# class StatisticsProcessor(DirectoriesProcessor):
#     def __init__(self, out_root_dir_path: str,
#                  is_raw_data: bool = True,
#                  block_duration_in_seconds: int = 60,
#                  block_delta_in_seconds: int = 15,
#                  packet_size_limit: int = 1500):
#
#         super().__init__(block_duration_in_seconds, block_delta_in_seconds, packet_size_limit)
#         self.out_root_dir = Path(out_root_dir_path)
#         self.is_raw_data = is_raw_data
#
#     def _process_dir_files(self, input_dir_path: Path):
#         out_dir_path = create_output_dir(self.out_root_dir, input_dir_path)
#         num_packets_hist_file_path = out_dir_path / 'num_packets_hist.png'
#         time_intervals_hist_file_path = out_dir_path / 'time_intervals_hist.png'
#         avg_size_dist_file_path = out_dir_path / 'avg_size_dist.png'
#         num_packets_cumulative_hist_file_path = out_dir_path / 'num_packets_cumulative_hist.txt'
#         time_intervals_cumulative_hist_file_path = out_dir_path / 'time_intervals_cumulative_hist.txt'
#         avg_size_distcumulative_hist_file_path = out_dir_path / 'avg_size_dist_cumulative_hist.txt'
#
#         blocks_meta = []
#         for file in get_dir_csvs(input_dir_path):
#             blocks_meta += self.process_file_to_flows(file)
#
#         num_packets, intervals, sizes = zip(*blocks_meta)
#         sizes = [s for sublist in sizes for s in sublist]
#
#         self._save_cumulative_hist_as_txt(num_packets, num_packets_cumulative_hist_file_path)
#         self._save_cumulative_hist_as_txt(intervals, time_intervals_cumulative_hist_file_path)
#         self._save_cumulative_hist_as_txt(sizes, avg_size_distcumulative_hist_file_path)
#
#         cap = self._get_numerical_cap(num_packets, percent=0.9)
#         packet_amount_bins = np.linspace(0, cap, 100)
#         time_interval_bins = np.linspace(0, 60, 100)
#         size_bins = np.linspace(0, 1500, 100)
#
#         self._save_hist_as_image(num_packets_hist_file_path, num_packets,
#                                  packet_amount_bins, label='num packets')
#
#         self._save_hist_as_image(time_intervals_hist_file_path, intervals,
#                                  time_interval_bins, label='time intervals')
#
#         self._save_hist_as_image(avg_size_dist_file_path, sizes,
#                                  size_bins, label='avg sizes')
#
#     def process_file_to_flows(self, input_file_path: Path):
#         file_extension = input_file_path.suffix
#         if file_extension != '.csv':
#             return []
#
#         print('processing ' + str(input_file_path))
#         with input_file_path.open(newline='') as f_in:
#             data = csv.reader(f_in, delimiter=',')
#
#             blocks = []
#             for row in data:
#                 if self.is_raw_data:
#                     flow = self.transform_row_to_flow(row)
#                     blocks += self.split_flow_to_blocks(flow)
#                 else:
#                     blocks += row
#
#             return list(map(self._get_block_meta, blocks))
#
#     @staticmethod
#     def _get_block_meta(block: List):
#         num_packets = block[0]
#         if num_packets != 0:
#             time_interval = int(round(block[num_packets+1])) - int(round(block[1]))
#         else:
#             time_interval = 0
#
#         sizes = block[num_packets+1:]
#
#         return num_packets, time_interval, sizes
#
#     @staticmethod
#     def _save_hist_as_image(file: Path, vals, bins, label: str):
#         plt.hist(vals, bins, alpha=0.5, label=label)
#         plt.legend(loc='upper right')
#         plt.savefig(file)
#         plt.clf()
#
#     @staticmethod
#     def _get_numerical_cap(a: Sequence[int], percent: float = 0.9):
#         sorted(a)
#         return a[int(percent * len(a))]
#
#     @staticmethod
#     def _save_cumulative_hist_as_txt(a: Sequence[int], file: Path):
#         a = sorted(a)
#         d = {}
#         current_val = a[0]
#         for i, x in enumerate(a):
#             if x != current_val:
#                 d[current_val] = i
#                 current_val = x
#         d[current_val] = len(a)
#
#         with file.open('w+') as file:
#             file.write(str(d))

