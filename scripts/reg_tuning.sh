#!/usr/bin/env bash

out_dir=reg_out
log_path=$out_dir/out.log

mkdir -p $out_dir

python experiments/hyper_tuning.py \
--data-dir data_reg \
--out-dir $out_dir \
--bs-train 128 \
--bs-test 256 \
--epochs 40 \
--print-every 40 \
--checkpoint-every 100 \
--hidden-dims 64 \
--filters-per-layer 10 20 \
--layers-per-block 1 \
--parallel 0 \
--verbose 0 \
--k 5 | tee $log_path
