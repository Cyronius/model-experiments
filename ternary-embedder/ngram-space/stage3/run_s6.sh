#!/bin/bash
# S6: best L1 (S5 lr_sparse 3e-5) extended to T2 bigrams and trigrams.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
W=training-v2/work/ngram-space/runs/S5_d1024_lam1e-3_lrs3e-5/best.pt
$PY -I ngram-space/train.py --arm S6_d1024_lam1e-3_lrs3e-5 --d 1024 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --warm $W --lam 1e-3 --lr_sparse 3e-5
