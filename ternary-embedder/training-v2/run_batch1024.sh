#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s training-v2 -p test_contrastive_cache.py
for family in datafirst predata; do
 for precision in float ternary; do
  prefix="";rates=(1e-5 4e-5)
  if [[ "$family" == predata ]]; then prefix="predata-";rates=(4e-5); fi
  for rate in "${rates[@]}"; do
  checkpoint="training-v2/work/batch1024/starts/$prefix$precision.pt"
  output="training-v2/runs/batch1024-$prefix$precision-$rate"
  resume=()
  if [[ -f "$output/completed.json" ]]; then continue; fi
  if [[ -f "$output/latest.json" ]]; then
   latest=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["file"])' "$output/latest.json")
   resume=(--resume "$output/$latest")
  fi
  python training-v2/pair_train.py \
   --data training-v2/work/data-first/capfit/train.jsonl \
   --dev training-v2/work/data-first/capfit/dev.jsonl \
   --checkpoint "$checkpoint" --out "$output" --expected-layers 2 \
   --epochs 2 --batch 1024 --micro-batch 128 --lr "$rate" --temperature .05 \
   --eval-every 200 --source-floor 2 --save-every 100 --save-seconds 120 \
   --device cuda --amp bf16 "${resume[@]}"
  python - "$output" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());latest=json.loads((p/'latest.json').read_text())
assert latest['step']==cfg['total_steps']
(p/'completed.json').write_text(json.dumps({'step':latest['step'],'precision':cfg['precision'],'layers':cfg['expected_layers']})+'\n')
PY
 done
done
done
