#!/usr/bin/env bash

python expiraments/split_experiment.py \
--run-name vc \
--data-dir data_vpn_overlap_split \
--epochs 60 \
--checkpoints temp \
--load-checkpoint True \
--checkpoint-every 1

