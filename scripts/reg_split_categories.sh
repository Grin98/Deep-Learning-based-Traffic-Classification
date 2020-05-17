#!/usr/bin/env bash

out_dir=reg_sp_out
log_path=$out_dir/out.log

mkdir -p $out_dir

python expiraments/split_experiment.py \
--data-dir data_reg_overlap_split \
--out-dir $out_dir \
--epochs 30 \
--checkpoints reg_overlap_split \
--load-checkpoint 0 \
--checkpoint-every 15 | tee $log_path