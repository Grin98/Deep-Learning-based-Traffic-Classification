#!/usr/bin/env bash

python expiraments/split_experiment.py \
--run-name vc \
--data-dir data_reg_overlap_split \
--epochs 60 \
--checkpoints reg_overlap_split \
--load-checkpoint False \
--checkpoint-every 30