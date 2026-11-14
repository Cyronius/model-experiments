#!/bin/bash
# L0 recipe (S2 d1024 lam 1e-3) at smaller sparse learning rates (STATUS 16:59).
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
for lr in 1e-4 3e-5; do
  $PY -I ngram-space/train.py --arm S2_d1024_lam1e-3_lrs$lr --d 1024 --lam 1e-3 --lr_sparse $lr
done
