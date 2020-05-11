#!/usr/bin/env bash

python expiraments/split_experiment.py \
--run-name vc \
--data-dir data_reg_overlap_split \
--epochs 60 \
--checkpoints temp \
--load-checkpoint False \
--checkpoint-every 30