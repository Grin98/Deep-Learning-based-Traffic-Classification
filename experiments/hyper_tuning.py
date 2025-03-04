import os
import random
import sys
from multiprocessing import Lock, current_process, set_start_method
from time import sleep

sys.path.append("../")
sys.path.append("./")

import itertools
from heapq import nlargest
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Sequence, Iterable, Tuple

import numpy as np
import torch

from experiments.cross_validation import CrossValidation
from experiments.experiment import Experiment
from misc.utils import create_dir
from misc.output import Logger


def list_chunk(l: Sequence, n: int) -> Tuple[Sequence, ...]:
    '''
    return a list of n even chunks of l
    '''
    sizes = np.full(n, len(l) // n)
    sizes[:len(l) % n] += 1
    ends = np.cumsum(sizes)

    return tuple(l[ends[i] - sizes[i]:ends[i]] for i in range(len(sizes)))


def conf_as_dict(conf: Sequence):
    return dict(lr=conf[0],
                reg=conf[1])


def run_experiments(experiment):
    confs, device = experiment
    torch.cuda.set_device(device)
    args = CrossValidation().parse_cli()
    out_file = Path(args.out_dir) / f'gpu{device}'
    with out_file.open(mode='w+') as f:
        log = Logger(file=f, verbose=False)
        return [run_conf(c, log) for c in confs]


def run_conf(conf, log: Logger):
    log.write(f'=== {current_process().pid} running config {conf} on gpu: {torch.cuda.current_device()} ===')
    cv = CrossValidation(log)
    args = cv.parse_cli()
    args = vars(args)
    args.update(conf)
    f1, _, _ = cv.run(**args)
    log.write(f'=== cv result for config {conf} on gpu: {torch.cuda.current_device()} is {f1} ===')
    return f1, conf


# python experiments/hyper_tuning.py --data-dir data_cv_reg --out-dir del --bs-train 128 --bs-test 256 --epochs 40 --save-checkpoint 0 --load-checkpoint 0 --checkpoint-every 100 --hidden-dims 64 --filters-per-layer 10 20 --layers-per-block 1 --parallel 0 --verbose 0 --k 5

if __name__ == '__main__':
    set_start_method('spawn')
    lr = list(np.logspace(start=-3, stop=-1, num=3))
    reg = list(np.logspace(start=-4, stop=-1, num=4))
    reg.append(0.0)

    print_lock = Lock()
    num_devices = torch.cuda.device_count()
    confs = list(map(conf_as_dict, itertools.product(lr, reg)))
    experiment = list(zip(list_chunk(confs, num_devices), range(num_devices)))
    print(experiment)
    pool = Pool(processes=num_devices)
    res = list(itertools.chain.from_iterable(pool.imap_unordered(run_experiments, experiment)))
    pool.close()
    pool.join()

    print('\n=== results ===')
    for r in res:
        print(f'{r}')

    best = nlargest(1, res, key=lambda r: r[0])[0]
    print(f'best config is {best[1]} with f1={best[0]}')
