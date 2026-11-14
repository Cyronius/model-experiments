#!/bin/bash
# S5 recipe with smaller sparse learning rates (see STATUS 16:56).
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
W=training-v2/work/ngram-space/runs/S4_d1024_s0.25_lam0/best.pt
for lr in 1e-4 3e-5; do
  $PY -I ngram-space/train.py --arm S5_d1024_lam1e-3_lrs$lr --d 1024 --n2 T1 --n3 T1 --ngram_init seed --scale 0.25 --warm $W --lam 1e-3 --lr_sparse $lr
done
