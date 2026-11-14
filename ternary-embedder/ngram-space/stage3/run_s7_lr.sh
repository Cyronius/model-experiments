#!/bin/bash
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
RUNS=training-v2/work/ngram-space/runs
$PY -I ngram-space/train.py --arm S7_d1024_lrm1e-4 --d 1024 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --warm $RUNS/S6_d1024_lam1e-3_lrs3e-5/best.pt --lam 1e-3 --lr_sparse 3e-5 --mlp --lr_mlp 1e-4
