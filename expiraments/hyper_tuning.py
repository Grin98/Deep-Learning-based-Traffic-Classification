import sys

sys.path.append("../")
sys.path.append("./")

import itertools
from heapq import nlargest
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from expiraments.cross_validation import CrossValidation
from expiraments.experiment import Experiment
from misc.utils import create_dir


def conf_as_dict(conf: Sequence):
    return dict(lr=conf[0],
                reg=conf[1]
                )


def run_conf(conf):
    cv = CrossValidation()
    args = cv.parse_cli()
    args = vars(args)
    args.update(conf)
    f1, _, _ = cv.run(**args)
    res = f1, conf
    return res


if __name__ == '__main__':
    run_conf({'lr': 1, 'reg': 2})
    # lr = list(np.logspace(start=-3, stop=-1, num=3))
    # reg = list(np.logspace(start=-4, stop=-1, num=4))
    # reg.append(0.0)
    #
    # confs = list(map(conf_as_dict, itertools.product(lr, reg)))
    # pool = Pool(processes=torch.cuda.device_count())
    # res = list(pool.imap_unordered(run_conf, confs))
    # print(f'results: {res}')
    # best = nlargest(1, res, key=lambda r: r[0])
    # print(f'best config is {res[1]} with f1={res[0]}')
