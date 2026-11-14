#!/bin/bash
# Stage 3, L1 bigram arms. $1 = width d, $2 = warm-start arm (selected S2).
# S3: + T1 bigrams, residuals zero.  S4: + T1 bigrams, residuals seeded x {0.25, 0.5, 1.0}.
# lambda in {0, 1e-4, 1e-3}; selection on validation loss.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
D=$1; W=training-v2/work/ngram-space/runs/$2/best.pt
for lam in 0 1e-4 1e-3; do
  $T --arm S3_d${D}_lam$lam --d $D --n2 T1 --ngram_init zero --warm $W --lam $lam
  for s in 0.25 0.5 1.0; do
    $T --arm S4_d${D}_s${s}_lam$lam --d $D --n2 T1 --ngram_init seed --scale $s --warm $W --lam $lam
  done
done
