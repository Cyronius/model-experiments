#!/bin/bash
# Stage 4 ladder on all rows, each rung warm-started from the one before,
# random chunk-grid offsets in training. L5 uses the c with the lower L4
# validation loss (rule fixed before results).
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
RUNS=training-v2/work/ngram-space/runs
C="--d 1024 --n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --lam 1e-3 --lr_sparse 3e-5 --lr_mlp 1e-4 --rows all --random_offset --resume --mlp"
$T --arm F_L3_c16 $C --warm $RUNS/F_L2/best.pt --chunk 16
$T --arm F_L4_c16 $C --warm $RUNS/F_L3_c16/best.pt --chunk 16 --K 1
$T --arm F_L4_c32 $C --warm $RUNS/F_L3_c16/best.pt --chunk 32 --K 1
best=$($PY -c "import json;a=json.load(open('$RUNS/F_L4_c16/train.json'))['best_val_loss'];b=json.load(open('$RUNS/F_L4_c32/train.json'))['best_val_loss'];print(16 if a<=b else 32)")
echo "L5 uses c=$best"
$T --arm F_L5_c$best $C --warm $RUNS/F_L4_c$best/best.pt --chunk $best --K 2
