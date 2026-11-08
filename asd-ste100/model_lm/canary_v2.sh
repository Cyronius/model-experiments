#!/usr/bin/env bash
# 24-row CPU canary for the constraint-v2 1.7B run (data/out/lm-ft-v1/ft-1.7b-c2.jsonl), against silent GPU
# faults. Detached launch:
#   setsid nohup model_lm/canary_v2.sh > data/out/lm-ft-v1/canary-v2.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-ft-v1/canary-v2.json exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
export PYTHONPATH=$PWD/data/out/lm-overlay
O=data/out/lm-ft-v1
[ -e $O/canary-cpu-v2.done ] || dibs run cpu --label "asd-ste100 constraint v2 CPU canary" --eta 40m -- $PY model_lm/generate.py \
  --model $O/lfm3-1.7b/best --out $O/canary-cpu-v2.jsonl --shots 0 --constrained --constraint v2 --limit 24 --inputs src \
  --device cpu --batch 8
$PY - <<'PYEOF'
import json
O = 'data/out/lm-ft-v1'
g = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/ft-1.7b-c2.jsonl')) if j['input'] == 'src'}
c = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/canary-cpu-v2.jsonl'))}
diff = [i for i in c if g.get(i) != c[i]]
json.dump(dict(rows=len(c), differ=len(diff), ids=diff), open(f'{O}/canary-v2.json', 'w'), indent=2)
print('canary v2: %d of %d CPU rows differ from the GPU run' % (len(diff), len(c)))
PYEOF
