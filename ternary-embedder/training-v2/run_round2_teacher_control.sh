#!/usr/bin/env bash
# Matched smaller-teacher control within the existing approved round-2 allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python training-v2/cache_teacher.py --teacher Qwen/Qwen3-Embedding-0.6B --download --plain-queries \
  --data training-v2/work/balanced-refined.jsonl --source training-v2/work/balanced-refined.source.json \
  --out training-v2/work/refined-plain06.npy --device cuda --batch 256 --batch-tokens 8192
python training-v2/rank_distill.py --data training-v2/work/balanced-refined.jsonl \
  --teacher training-v2/work/refined-plain06.npy --out training-v2/runs/balanced-plain06-anchor1.0 \
  --checkpoint tern-embed/pulls/grid1/runs/2a_384/model.pt \
  --steps 4000 --batch 32 --lr 2e-5 --float-fraction 0 --ramp-fraction 0 \
  --hard-negatives --qat-continuation --anchor-weight 1.0 --device cuda
# Isolate verified-positive supervision from the same prompted 8B / anchor-1 control.
python training-v2/rank_distill.py --data training-v2/work/balanced-refined.jsonl \
  --teacher training-v2/work/refined-prompted.npy --out training-v2/runs/balanced-prompted-anchor1.0-positive.2 \
  --checkpoint tern-embed/pulls/grid1/runs/2a_384/model.pt \
  --steps 4000 --batch 32 --lr 2e-5 --float-fraction 0 --ramp-fraction 0 \
  --hard-negatives --qat-continuation --anchor-weight 1.0 --positive-weight .2 --device cuda
# Recover float-control geometry while keeping the student fully quantized.
python training-v2/rank_distill.py --data training-v2/work/balanced-refined.jsonl \
  --teacher training-v2/work/refined-prompted.npy --out training-v2/runs/balanced-prompted-floatanchor1.0-positive.2 \
  --checkpoint tern-embed/pulls/grid1/runs/2a_384/model.pt \
  --anchor-checkpoint retrieval-diagnostics/float-control/runs/float150/model.pt \
  --steps 4000 --batch 32 --lr 2e-5 --float-fraction 0 --ramp-fraction 0 \
  --hard-negatives --qat-continuation --anchor-weight 1.0 --positive-weight .2 --device cuda
