#!/usr/bin/env bash
# Few-shot STE rewrites of the 679 doc dev pairs by LFM3-1.7B and LFM2.5-350M, constrained and free,
# on the local GPU under `dibs gpu`, then a CPU canary (first 24 rows, LFM3 constrained, fp32) to catch
# the silent GPU faults seen since 2026-11-05, then scoring. Detached launch:
#   setsid nohup model_lm/run_zero_shot.sh > data/out/lm-zeroshot-v1/run.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-zeroshot-v1/ALL.done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
# transformers 5.22.2 + tokenizers 0.23 overlay: LFM3 remote code needs newer transformers than the env has
export PYTHONPATH=$PWD/data/out/lm-overlay
VPY=data/out/.venv/bin/python
O=data/out/lm-zeroshot-v1
mkdir -p $O
gen() {  # name model flags...
  local name=$1 model=$2; shift 2
  [ -e $O/$name.done ] && return 0
  dibs run gpu --label "asd-ste100 LM few-shot $name" --eta 40m -- $PY model_lm/generate.py --model $model --out $O/$name.jsonl "$@"
}
gen lfm3-1.7b-c data/out/lfm/LFM3-1.7B --constrained
gen lfm2.5-350m-c data/out/lfm/LFM2.5-350M --constrained
gen lfm3-1.7b-free data/out/lfm/LFM3-1.7B
gen lfm2.5-350m-free data/out/lfm/LFM2.5-350M
if [ ! -e $O/canary-cpu.done ]; then
  dibs run cpu --label "asd-ste100 LM CPU canary" --eta 30m -- $PY model_lm/generate.py --model data/out/lfm/LFM3-1.7B \
    --out $O/canary-cpu.jsonl --constrained --limit 24 --inputs src --device cpu --batch 8
fi
$PY - <<'PYEOF'
import json
O='data/out/lm-zeroshot-v1'
g={j['id']:j['output'] for j in map(json.loads,open(f'{O}/lfm3-1.7b-c.jsonl')) if j['input']=='src'}
c={j['id']:j['output'] for j in map(json.loads,open(f'{O}/canary-cpu.jsonl'))}
diff=[i for i in c if g.get(i)!=c[i]]
json.dump(dict(rows=len(c),differ=len(diff),ids=diff),open(f'{O}/canary.json','w'),indent=2)
print('canary: %d of %d CPU rows differ from the GPU run' % (len(diff),len(c)))
PYEOF
env -u PYTHONPATH $VPY model_lm/score.py $O/scores.json $O/lfm3-1.7b-c.jsonl $O/lfm2.5-350m-c.jsonl $O/lfm3-1.7b-free.jsonl $O/lfm2.5-350m-free.jsonl
touch $O/ALL.done
