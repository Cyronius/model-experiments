#!/usr/bin/env bash
# Whole GPU job; idempotent so dibs --requeue can rerun it after a yield.
# train (resumable, rules.json max_steps) -> evaluate. Writes ../results/xattn-main/run.done.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
RES=../results
WORK=../../target/context-memory/xattn
if [ ! -f $WORK/ckpt/xattn-main/final.pt ]; then
  $PY train.py train --output $RES/xattn-main
fi
if [ ! -f $RES/xattn-main/eval/eval.done ]; then
  $PY train.py evaluate --output $RES/xattn-main/eval --checkpoint $WORK/ckpt/xattn-main/final.pt
fi
touch $RES/xattn-main/run.done
