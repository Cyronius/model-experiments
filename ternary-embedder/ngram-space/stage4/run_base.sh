#!/bin/bash
# Stage 4 base: the selected Stage 3 recipe on all 1,173,363 training texts.
# F_L1: C_L0_all + T2 bigram/trigram seeds x 0.25, lam 1e-3, lr_sparse 3e-5.
# F_L2: F_L1 + readout MLP (lr_mlp 1e-4), for B2 at L2 on all rows.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
RUNS=training-v2/work/ngram-space/runs
C="--d 1024 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --lam 1e-3 --lr_sparse 3e-5 --rows all --resume"
$T --arm F_L1 $C --warm $RUNS/C_L0_all/best.pt
$T --arm F_L2 $C --warm $RUNS/F_L1/best.pt --mlp --lr_mlp 1e-4
