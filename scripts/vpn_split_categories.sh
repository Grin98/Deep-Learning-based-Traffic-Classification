#!/usr/bin/env bash

python expiraments/split_experiment.py \
--data-dir data_vpn_overlap_split \
--epochs 15 \
--checkpoints vpn_overlap_split \
--load-checkpoint 0 \
--checkpoint-every 15



