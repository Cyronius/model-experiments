#!/bin/bash
# Stage 3, S5: best bigram arm + T1 trigrams, same initialization choice.
# $1 = d, $2 = warm arm (best bigram arm), $3 = ngram_init (seed|zero), $4 = scale
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
D=$1; W=training-v2/work/ngram-space/runs/$2/best.pt
for lam in 0 1e-4 1e-3; do
  $T --arm S5_d${D}_lam$lam --d $D --n2 T1 --n3 T1 --ngram_init $3 --scale $4 --warm $W --lam $lam
done
