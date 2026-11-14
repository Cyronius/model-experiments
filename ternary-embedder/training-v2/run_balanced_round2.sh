#!/usr/bin/env bash
# A NEW paid job requires separate approval. This script never rents a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ ! -s training-v2/work/balanced.jsonl || ! -s training-v2/work/balanced.source.json ]]; then
  python training-v2/fetch_balanced.py --rows 50000 --out training-v2/work/balanced.jsonl
fi
python training-v2/cache_teacher.py --teacher Qwen/Qwen3-Embedding-8B --download \
  --data training-v2/work/balanced.jsonl --source training-v2/work/balanced.source.json \
  --out training-v2/work/balanced-prompted.npy --device cuda --batch 128 --batch-tokens 4096
python training-v2/cache_teacher.py --teacher Qwen/Qwen3-Embedding-8B --plain-queries \
  --reuse-docs training-v2/work/balanced-prompted.npy \
  --data training-v2/work/balanced.jsonl --source training-v2/work/balanced.source.json \
  --out training-v2/work/balanced-plain.npy --device cuda --batch 128 --batch-tokens 4096
for config in 'prompted .1' 'prompted 1.0' 'plain 1.0'; do
  read -r teacher anchor <<< "$config"
  python training-v2/rank_distill.py --data training-v2/work/balanced.jsonl \
    --teacher "training-v2/work/balanced-$teacher.npy" --out "training-v2/runs/balanced-$teacher-anchor$anchor" \
    --checkpoint tern-embed/pulls/grid1/runs/2a_384/model.pt \
    --steps 4000 --batch 32 --lr 2e-5 --float-fraction 0 --ramp-fraction 0 \
    --hard-negatives --qat-continuation --anchor-weight "$anchor" --device cuda
done
