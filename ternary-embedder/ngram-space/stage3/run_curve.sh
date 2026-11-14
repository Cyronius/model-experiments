#!/bin/bash
# Stage 3 data curve: the best L1 recipe (S5) at $1 = 600k | all training rows.
# L0 seeded unigrams (d 1024, lam 1e-3), then + T1 bigrams and trigrams seeded
# x 0.25, warm, lam 1e-3 (early stopping keeps epoch 0 if training does not help).
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
R=$1; RUNS=training-v2/work/ngram-space/runs
$T --arm C_L0_$R --d 1024 --lam 1e-3 --rows $R
$T --arm C_L1_$R --d 1024 --n2 T1 --n3 T1 --ngram_init seed --scale 0.25 --warm $RUNS/C_L0_$R/best.pt --lam 1e-3 --rows $R
