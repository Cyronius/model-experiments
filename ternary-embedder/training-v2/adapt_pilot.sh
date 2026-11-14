#!/usr/bin/env bash
# Extend the SAME pilot's three arms to resolve the observed short-QAT failure.
# No additional data, teachers, GPU allocation or budget increase.
set -euo pipefail
cd /workspace/ternary-pilot/ternary-embedder
trap 'echo failed > training-v2/pilot.status' ERR
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf running > training-v2/pilot.status
for config in '0.6B gradual .25' '8B gradual .25' '8B abrupt 0'; do
  read -r size schedule ramp <<< "$config"
  python training-v2/rank_distill.py --data training-v2/work/pilot.jsonl \
    --teacher "training-v2/work/pilot-teacher-$size.npy" --out "training-v2/runs/adapted-$size-$schedule" \
    --steps 4000 --batch 32 --lr 2e-5 --float-fraction .5 --ramp-fraction "$ramp" \
    --hard-negatives --restart-qat-lr --device cuda
done
printf done > training-v2/pilot.status
