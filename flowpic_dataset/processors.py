import itertools
import random
import csv
from math import ceil
from typing import Sequence, NamedTuple
from abc import ABC, abstractmethod

from misc.constants import PACKET_SIZE_LIMIT, BLOCK_DELTA, BLOCK_DURATION
from misc.data_classes import Flow, Block
from misc.utils import *


class BasicProcessor:
    """
    provides the basic processing functions of flows
    """

    def process_file_to_flows(self, input_file_path: Path):
        file_extension = input_file_path.suffix
        if file_extension != '.csv':
            return []

        with input_file_path.open(newline='') as f_in:
            data = csv.reader(f_in, delimiter=',')
            return [Flow.create_from_row(row) for row in data]

    def split_multiple_flows_to_blocks(self, flows: Sequence[Flow]) -> Sequence[Block]:
        blocks = []
        for f in flows:
            blocks += self.split_flow_to_blocks(f)

        return blocks

    def split_flow_to_blocks(self, flow: Flow) -> Sequence[Block]:
        times = flow.times
        sizes = flow.sizes
        num_blocks = max(int(times[-1] / BLOCK_DELTA - BLOCK_DURATION / BLOCK_DELTA) + 1, 1)

        blocks = []
        for b in range(num_blocks):
            start = b * BLOCK_DELTA
            end = b * BLOCK_DELTA + BLOCK_DURATION

            mask = ((times >= start) & (times < end))
            if np.count_nonzero(mask) == 0:
                continue

            block_times = times[mask]
            block_sizes = sizes[mask]
            num_packets = len(block_times)

            # normalize times to start from 0
            block_start_time = b * BLOCK_DELTA
            block_times = block_times - block_start_time

            block = Block(block_start_time, num_packets, list(zip(list(block_times), list(block_sizes))))
            blocks.append(block)

        return blocks

    @staticmethod
    def split(flows: Sequence[Flow], group_percents: Sequence[float]) -> Sequence[Sequence[Flow]]:
        if sum(group_percents) != 1.0:
            raise Exception('sum of percents of group must be equal to 1')

        total_num_flows = len(flows)
        random.shuffle(flows)
        groups: List[List] = []
        for p in group_percents:
            g_len = int(p * total_num_flows)
            g = flows[:g_len]
            flows = flows[g_len:]
            groups.append(g)

        for i, f in enumerate(flows):
            groups[i % len(groups)].append(f)

        return groups

    def sample_group_blocks(self, groups: Sequence[Sequence[Block]], target_amount: int):
        current_amount = sum(map(lambda g: len(g), groups))
        excess = current_amount - target_amount
        if excess <= 0:
            return groups

        groups = sorted(groups, key=lambda g: len(g), reverse=True)
        for i in range(len(groups)):
            group_diff = (len(groups[i]) - len(groups[i + 1])) if (i + 1) != len(groups) else ceil(excess / (i + 1))
            diff = min(group_diff, ceil(excess / (i + 1)))
            target_amount = len(groups[i]) - diff
            groups = [self.sample_blocks(g, target_amount) if j <= i else g for j, g in enumerate(groups)]

            excess -= diff * (i + 1)
            if excess <= 0:
                break

        return groups

    @staticmethod
    def sample_blocks(blocks: Sequence[Block], target_amount: int):
        current_amount = len(blocks)
        if current_amount > target_amount:
            indices = random.sample(range(current_amount), target_amount)
            blocks = create_array_of_objects(blocks)
            blocks = blocks[indices]

        return blocks

    @staticmethod
    def _write_blocks(blocks: Sequence[Block], file: Path, tag: str = None):
        tag = '' if tag is None else tag
        print(f'{tag} saving {len(blocks)}')
        blocks = [b.convert_to_row() for b in blocks]
        with file.open('w+', newline='') as out:
            csv.writer(out, delimiter=',').writerows(blocks)
        return len(blocks)


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


class HoldOutPreProcessor(DirectoriesProcessor):
    """
    statically splits dataset of flows to Train and Test sets before splitting each flow to blocks
    and in addition there is an overlap between consecutive blocks depending on the value of [block_delta_in_seconds]
    """

    def __init__(self, out_root_dir,
                 test_percent: float,
                 train_size_cap: int,
                 test_size_cap: int,
                 ):
        super().__init__()
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
        flows = [self.process_file_to_flows(file) for file in get_dir_csvs(input_dir_path)]
        flows = list(itertools.chain.from_iterable(flows))

        train_flows, test_flows = self.split(flows, [1 - self.test_percent, self.test_percent])
        train_blocks = self.split_multiple_flows_to_blocks(train_flows)
        test_blocks = self.split_multiple_flows_to_blocks(test_flows)

        train_blocks = self.sample_blocks(train_blocks, target_amount=self.train_size_cap)
        test_blocks = self.sample_blocks(test_blocks, target_amount=self.test_size_cap)

        self._write_blocks(train_blocks, train_file_path, tag='train')
        self._write_blocks(test_blocks, test_file_path, tag='test')


class CrossValidationPreProcessor(DirectoriesProcessor):

    def __init__(self,
                 out_root_dir,
                 test_percent: float,
                 train_size_cap: int,
                 test_size_cap: int,
                 k: int
                 ):
        """
        :param out_root_dir: dir path to write to the processed dataset
        :param test_percent: percent of flows (not blocks) to be saved as test
        :param train_size_cap: cap of how many blocks each category in train will have
        :param test_size_cap: cap of how many blocks each category in train will have
        :param k: num of folds
        :param block_duration_in_seconds: the duration of a block
        :param block_delta_in_seconds: the time difference between the start of 2 consecutive blocks
        :param packet_size_limit: packets with larger size then this parameter will be discarded
        """
        super().__init__()
        self.out_root_dir = Path(out_root_dir)
        self.train_path = out_root_dir / Path('train')
        self.test_path = out_root_dir / Path('test')
        self.groups = [self.train_path / f'g{i}' for i in range(k)]
        self.test_percent = test_percent
        self.train_size_cap = train_size_cap
        self.test_size_cap = test_size_cap
        self.k = k

    def _process_dir_files(self, input_dir_path: Path):
        out_file = 'data.csv'
        group_out_file_paths = self.create_out_file_paths(self.groups, input_dir_path, out_file)
        test_file_path = create_output_dir(self.test_path, input_dir_path) / out_file

        print('processing %s' % input_dir_path)
        flows = [self.process_file_to_flows(file) for file in get_dir_csvs(input_dir_path)]
        flows = list(itertools.chain.from_iterable(flows))

        train, test = self.split(flows, [1 - self.test_percent, self.test_percent])
        groups = self.split(train, [1.0 / self.k] * self.k)

        groups_blocks = [self.split_multiple_flows_to_blocks(g) for g in groups]
        test_blocks = self.split_multiple_flows_to_blocks(test)

        groups_blocks = self.sample_group_blocks(groups_blocks, target_amount=self.train_size_cap)
        test_blocks = self.sample_blocks(test_blocks, target_amount=self.test_size_cap)

        total = 0
        for i, f in enumerate(group_out_file_paths):
            total += self._write_blocks(groups_blocks[i], f, tag=f'group{i}')
        print(f'saved in total {total}')

        self._write_blocks(test_blocks, test_file_path, tag='test')

    @staticmethod
    def create_out_file_paths(out_root_dirs: Sequence[Path], input_dir_path: Path, file_name: str):
        dir_paths = create_multiple_output_dirs(out_root_dirs, input_dir_path)
        return [p / file_name for p in dir_paths]


class QuickFlowFileProcessor:
    """
    splits file flows to raw blocks but doesn't write them out to a file,
    instead it returns them.
    """

    def __init__(self):
        self.p: BasicProcessor = BasicProcessor()

    def transform_file_to_blocks(self, file: Path) -> Sequence[Block]:
        flows = self.p.process_file_to_flows(file)
        return self.p.split_multiple_flows_to_blocks(flows)
