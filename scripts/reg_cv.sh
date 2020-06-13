#!/usr/bin/env bash

out_dir=reg_out
log_path=$out_dir/out.log

mkdir -p $out_dir

python expiraments/CrossValidation.py \
--data-dir data_cv_reg \
--out-dir $out_dir \
--bs-train 128 \
--bs-test 256 \
--epochs 40 \
--lr 0.001 \
--save-checkpoint 1 \
--load-checkpoint 1 \
--checkpoint-every 20 \
--hidden-dims 64 \
--filters-per-layer 10 20 \
--layers-per-block 1 \
--k 5 | tee $log_path