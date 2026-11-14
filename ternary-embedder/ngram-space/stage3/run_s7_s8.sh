#!/bin/bash
# S7: best L1 (S6) + readout MLP. S8: the S7 recipe at d 512.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
RUNS=training-v2/work/ngram-space/runs
$T --arm S7_d1024 --d 1024 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --warm $RUNS/S6_d1024_lam1e-3_lrs3e-5/best.pt --lam 1e-3 --lr_sparse 3e-5 --mlp
$T --arm S8a_d512_L1 --d 512 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --warm $RUNS/S2_d512_lam1e-3/best.pt --lam 1e-3 --lr_sparse 3e-5
$T --arm S8_d512 --d 512 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --warm $RUNS/S8a_d512_L1/best.pt --lam 1e-3 --lr_sparse 3e-5 --mlp
