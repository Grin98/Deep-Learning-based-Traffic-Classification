#!/usr/bin/env bash

python expiraments/split_experiment.py \
--data-dir data_reg_overlap_split \
--epochs 60 \
--checkpoints reg_overlap_split \
--load-checkpoint 0 \
--checkpoint-every 30