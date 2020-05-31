#!/usr/bin/env bash

out_dir=reg_out
log_path=$out_dir/out.log

mkdir -p $out_dir

python expiraments/split_experiment.py \
--data-dir data_reg \
--out-dir $out_dir \
--epochs 35 \
--lr 0.002 \
--checkpoints reg \
--load-checkpoint 0 \
--checkpoint-every 35 | tee $log_path