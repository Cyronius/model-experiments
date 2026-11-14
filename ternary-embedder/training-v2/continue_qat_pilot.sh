#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ternary-pilot/ternary-embedder
trap 'echo failed > training-v2/pilot.status' ERR
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf running > training-v2/pilot.status
python training-v2/rank_distill.py --data training-v2/work/pilot.jsonl \
  --teacher training-v2/work/pilot-teacher-8B.npy --out training-v2/runs/continued-8B-qat \
  --checkpoint tern-embed/pulls/grid1/runs/2a_384/model.pt \
  --steps 4000 --batch 32 --lr 2e-5 --float-fraction 0 --ramp-fraction 0 \
  --hard-negatives --qat-continuation --device cuda
printf done > training-v2/pilot.status
