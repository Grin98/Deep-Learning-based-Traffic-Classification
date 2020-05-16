#!/usr/bin/env bash

out_dir=bi_out/vid
log_path=$out_dir/vid_sp.log

mkdir -p $out_dir

python expiraments/split_experiment.py \
--data-dir data_bi \
--data-out $out_dir \
--epochs 15 \
--checkpoints bi_vid \
--load-checkpoint 0 | tee $log_path