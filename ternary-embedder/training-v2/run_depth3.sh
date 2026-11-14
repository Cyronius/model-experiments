#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for precision in float ternary; do
  for depth in 2 3; do
    checkpoint="training-v2/work/depth3/starts/$depth-$precision.pt"
    output="training-v2/runs/depth3-$depth-$precision"
    resume=();growth=()
    if [[ "$depth" == 3 ]]; then growth=(--growth-warmup-steps 500); fi
    if [[ -f "$output/completed.json" ]]; then continue; fi
    if [[ -f "$output/latest.json" ]]; then
      latest=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["file"])' "$output/latest.json")
      resume=(--resume "$output/$latest")
    fi
    python training-v2/pair_train.py \
      --data training-v2/work/data-first/capfit/train.jsonl \
      --dev training-v2/work/data-first/capfit/dev.jsonl \
      --checkpoint "$checkpoint" --out "$output" --expected-layers "$depth" \
      --epochs 2 --batch 128 --lr 1e-5 --temperature .05 \
      --eval-every 1500 --source-floor 2 --save-every 500 --save-seconds 120 \
      --device cuda --amp bf16 "${growth[@]}" "${resume[@]}"
    python - "$output" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());latest=json.loads((p/'latest.json').read_text())
assert latest['step']==cfg['total_steps']
(p/'completed.json').write_text(json.dumps({'step':latest['step'],'precision':cfg['precision'],'layers':cfg['expected_layers']})+'\n')
PY
  done
done
