#!/bin/bash
# Stage 3, L1 bigram arms trained from the seeds (no warm start), same grid as
# run_l1a.sh. Added after the warm-started arms never improved on their start
# (fresh optimizer state made epoch 1 worse; early stopping kept epoch 0).
# $1 = width d. Arm names end in _cold.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
D=$1
for lam in 1e-3 1e-4 0; do
  $T --arm S3_d${D}_lam${lam}_cold --d $D --n2 T1 --ngram_init zero --lam $lam
  for s in 0.25 0.5 1.0; do
    $T --arm S4_d${D}_s${s}_lam${lam}_cold --d $D --n2 T1 --ngram_init seed --scale $s --lam $lam
  done
done
