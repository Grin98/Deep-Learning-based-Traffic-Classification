#!/usr/bin/env bash

out_dir=vpn_sp_out
log_path=$out_dir/vpn_sp.log

python expiraments/split_experiment.py \
--data-dir data_vpn_overlap_split \
--out-dir $out_dir \
--epochs 15 \
--checkpoints vpn_overlap_split \
--load-checkpoint 0 \
--checkpoint-every 15 | tee $log_path
