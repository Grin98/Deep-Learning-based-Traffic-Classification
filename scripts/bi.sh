#!/usr/bin/env bash

out_dir=bi_out/vid
log_path=$out_dir/out.log

mkdir -p $out_dir

python expiraments/split_experiment.py \
--data-dir data_bi \
--out-dir $out_dir \
--epochs 10 \
--checkpoints bi \
--load-checkpoint 0 | tee $log_path