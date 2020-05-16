#!/usr/bin/env bash

out_dir=reg_sp_out
log_path=$out_dir/reg_sp.log

python expiraments/split_experiment.py \
--data-dir data_reg_overlap_split \
--out-dir $out_dir \
--epochs 60 \
--checkpoints reg_overlap_split \
--load-checkpoint 0 \
--checkpoint-every 30 | tee $log_path