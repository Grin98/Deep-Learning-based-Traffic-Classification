#!/usr/bin/env bash

out_dir=reg_out
log_path=$out_dir/out.log

mkdir -p $out_dir

python experiments/split_experiment.py \
--data-dir data_cv_nz_reg \
--out-dir $out_dir \
--data-format 1 \
--bs-train 128 \
--bs-test 256 \
--epochs 40 \
--print-every 5 \
--lr 0.001 \
--reg 0.0001 \
--checkpoint-every 20 \
--hidden-dims 64 \
--filters-per-layer 10 20 \
--layers-per-block 1 \
--parallel 1 \
--verbose 1 | tee $log_path
