#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for precision in float ternary; do
  checkpoint=retrieval-diagnostics/float-control/runs/float150/model.pt
  if [[ "$precision" == ternary ]]; then checkpoint=tern-embed/pulls/grid1/runs/2a_384/model.pt; fi
  output="training-v2/runs/data-first-$precision"
  resume=()
  if [[ -f "$output/latest.json" ]]; then
    latest=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["file"])' "$output/latest.json")
    resume=(--resume "$output/$latest")
  fi
  if [[ -f "$output/completed.json" ]]; then continue; fi
  python training-v2/pair_train.py \
    --data training-v2/work/data-first/capfit/train.jsonl \
    --dev training-v2/work/data-first/capfit/dev.jsonl \
    --checkpoint "$checkpoint" --out "$output" \
    --epochs 2 --batch 128 --lr 1e-5 --temperature .05 \
    --eval-every 1500 --source-floor 2 --save-every 500 --save-seconds 120 \
    --device cuda --amp bf16 "${resume[@]}"
  python - "$output" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());latest=json.loads((p/'latest.json').read_text())
assert latest['step']==cfg['total_steps'], 'Training stopped before its budget completed'
(p/'completed.json').write_text(json.dumps({'step':latest['step'],'precision':cfg['precision']})+'\n')
PY
done
