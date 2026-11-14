#!/usr/bin/env bash
# Workload only: no pod allocation. Run within an explicitly approved time/cost cap.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
"$PYTHON" training-v2/fetch_training.py --rows 4096 --out training-v2/work/pilot.jsonl
for size in 0.6B 8B; do
  batch=128
  if [[ "$size" == 8B ]]; then batch=4; fi
  "$PYTHON" training-v2/cache_teacher.py --teacher "Qwen/Qwen3-Embedding-$size" --download \
    --data training-v2/work/pilot.jsonl --source training-v2/work/pilot.source.json \
    --out "training-v2/work/pilot-teacher-$size.npy" --device cuda --batch "$batch"
  "$PYTHON" training-v2/rank_distill.py --data training-v2/work/pilot.jsonl \
    --teacher "training-v2/work/pilot-teacher-$size.npy" --out "training-v2/runs/pilot-$size-gradual" \
    --steps 400 --batch 32 --lr 2e-5 --float-fraction .5 --ramp-fraction .25 --hard-negatives --device cuda
done
"$PYTHON" training-v2/rank_distill.py --data training-v2/work/pilot.jsonl \
  --teacher training-v2/work/pilot-teacher-8B.npy --out training-v2/runs/pilot-8B-abrupt \
  --steps 400 --batch 32 --lr 2e-5 --float-fraction .5 --ramp-fraction 0 --hard-negatives --device cuda
