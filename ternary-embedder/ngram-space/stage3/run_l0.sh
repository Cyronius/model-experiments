#!/bin/bash
# Stage 3, L0 arms: S0 (seeds, no training), S1 (random init, d1024), S2 (seeds, d512 and d1024).
# lambda in {0, 1e-4, 1e-3} for every trained arm; selection on validation loss.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
$T --arm S0 --d 1024 --epochs 0
for lam in 0 1e-4 1e-3; do
  $T --arm S1_d1024_lam$lam --d 1024 --init random --lam $lam
  $T --arm S2_d1024_lam$lam --d 1024 --lam $lam
  $T --arm S2_d512_lam$lam --d 512 --lam $lam
done
