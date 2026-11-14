#!/bin/bash
# Stage 4 variants of the selected rung (L4, c = 32), rules fixed before results:
#  F_L4_c32_d512  the other width: warm from the best d512 L1 (S8a, 300k rows),
#                 grown to L4 c32 (MLP + chunking + 1 mixer layer) in one run on all rows.
#  F_L4_c32_ctrl  the control: no warm start, Stage 2 seeds (unigram + T2 n-grams x 0.25),
#                 L4 c32 trained in a single run. Same flags as the ladder, including
#                 lr_sparse 3e-5 (tuned for warm-started n-gram rows; stated in RESULTS).
#  F_L5_c32_clip  deviation, not in the plan: L5 diverged in epoch 1 without gradient
#                 clipping; retry from F_L4_c32 with grad-norm clip 1.0 on dense params.
#                 It enters B2 as an L5 arm only if it beats F_L4_c32 on validation loss.
set -e
cd /home/cyrus/code/model-experiments/ternary-embedder
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
T="$PY -I ngram-space/train.py"
RUNS=training-v2/work/ngram-space/runs
C="--n2 T2 --n3 T2 --ngram_init seed --scale 0.25 --lam 1e-3 --lr_sparse 3e-5 --lr_mlp 1e-4 --rows all --random_offset --resume --mlp --chunk 32"
done_() { grep -q best_train_loss $RUNS/$1/train.json 2>/dev/null; }  # finished arms are skipped on rerun
done_ F_L4_c32_d512 || $T --arm F_L4_c32_d512 $C --d 512 --warm $RUNS/S8a_d512_L1/best.pt --K 1
done_ F_L4_c32_ctrl || $T --arm F_L4_c32_ctrl $C --d 1024 --init seed --K 1
done_ F_L5_c32_clip || $T --arm F_L5_c32_clip $C --d 1024 --warm $RUNS/F_L4_c32/best.pt --K 2 --clip 1.0
