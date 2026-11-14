#!/usr/bin/env bash
# Run inside a separately approved, bounded GPU allocation. Does not rent hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
# Install dependencies and stage the existing float checkpoint + tokenizer first.
# Fetch Qwen at the pinned revision into cache/hub before running.
"$PYTHON" training-v2/fetch_training.py --rows 50000 --out training-v2/work/train.jsonl
"$PYTHON" training-v2/cache_teacher.py --data training-v2/work/train.jsonl \
  --source training-v2/work/train.source.json --out training-v2/work/teacher.npy --device cuda --batch 128
for arm in rank_abrupt hard_abrupt hard_gradual; do
  extra=()
  if [[ "$arm" != rank_abrupt ]]; then extra+=(--hard-negatives); fi
  ramp=0
  if [[ "$arm" == hard_gradual ]]; then ramp=.25; fi
  "$PYTHON" training-v2/rank_distill.py --data training-v2/work/train.jsonl \
    --teacher training-v2/work/teacher.npy --out "training-v2/runs/$arm" \
    --steps 4000 --batch 64 --lr 2e-5 --float-fraction .5 --ramp-fraction "$ramp" \
    --device cuda "${extra[@]}"
done
