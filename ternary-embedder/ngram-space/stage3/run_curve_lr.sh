#!/bin/bash
# Data curve rerun (STATUS 16:59 rule): L1 step with the selected lr_sparse 3e-5,
# warm from the curve's L0 at the same row count. $1 = 600k | all.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
R=$1; RUNS=training-v2/work/ngram-space/runs
$PY -I ngram-space/train.py --arm C_L1lr_$R --d 1024 --n2 T1 --n3 T1 --ngram_init seed --scale 0.25 --warm $RUNS/C_L0_$R/best.pt --lam 1e-3 --lr_sparse 3e-5 --rows $R
