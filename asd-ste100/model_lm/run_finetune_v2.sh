#!/usr/bin/env bash
# Fine-tune round 2: the round-1 recipe (run_finetune.sh: 3 epochs, LR 3e-5 / 1e-5, epoch by held-out loss)
# plus identity pairs (finetune.py --identity: each reference rewritten to itself), so the models learn to
# leave text that is already STE unchanged. Both models; dev rewrites with constraint v2 and free; a 24-row
# CPU canary for each constrained run; scores beside round 1's v2 runs. Detached launch:
#   setsid nohup model_lm/run_finetune_v2.sh > data/out/lm-ft-v2/run.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-ft-v2/ALL.done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
export PYTHONPATH=$PWD/data/out/lm-overlay
VPY=data/out/.venv/bin/python
O=data/out/lm-ft-v2
R1=data/out/lm-ft-v1
mkdir -p $O
ft() {  # name model lr eta; retried: a non-finite loss from a GPU fault stops finetune.py, which resumes after the last saved epoch
  local try
  for try in 1 2 3; do
    [ -e $O/$1/FT.done ] && return 0
    dibs run gpu --label "asd-ste100 LM fine-tune round 2 $1" --eta $4 -- $PY model_lm/finetune.py --model $2 --out $O/$1 \
      --lr $3 --epochs 3 --identity && return 0
    echo "fine-tune $1 attempt $try failed" >&2
  done
  return 1
}
gen() {  # name model flags...
  local name=$1 model=$2; shift 2
  [ -e $O/$name.done ] && return 0
  dibs run gpu --label "asd-ste100 LM round 2 eval $name" --eta 30m -- $PY model_lm/generate.py --model $model --out $O/$name.jsonl --shots 0 "$@"
}
canary() {  # name model
  [ -e $O/canary-$1.done ] || dibs run cpu --label "asd-ste100 round 2 CPU canary $1" --eta 30m -- $PY model_lm/generate.py \
    --model $2 --out $O/canary-$1.jsonl --shots 0 --constrained --constraint v2 --limit 24 --inputs src --device cpu --batch 8
  $PY - $1 <<'PYEOF'
import json, sys
O, n = 'data/out/lm-ft-v2', sys.argv[1]
g = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/{n}.jsonl')) if j['input'] == 'src'}
c = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/canary-{n}.jsonl'))}
diff = [i for i in c if g.get(i) != c[i]]
json.dump(dict(rows=len(c), differ=len(diff), ids=diff), open(f'{O}/canary-{n}.json', 'w'), indent=2)
print('canary %s: %d of %d CPU rows differ from the GPU run' % (n, len(diff), len(c)))
PYEOF
}
ft lfm2.5-350m data/out/lfm/LFM2.5-350M 3e-5 75m
gen r2-350m-c2 $O/lfm2.5-350m/best --constrained --constraint v2
gen r2-350m-free $O/lfm2.5-350m/best
canary r2-350m-c2 $O/lfm2.5-350m/best
ft lfm3-1.7b data/out/lfm/LFM3-1.7B 1e-5 5h
gen r2-1.7b-c2 $O/lfm3-1.7b/best --constrained --constraint v2
gen r2-1.7b-free $O/lfm3-1.7b/best
canary r2-1.7b-c2 $O/lfm3-1.7b/best
env -u PYTHONPATH $VPY model_lm/score.py $O/scores.json $R1/ft-350m-c2.jsonl $O/r2-350m-c2.jsonl $R1/ft-350m-free.jsonl \
  $O/r2-350m-free.jsonl $R1/ft-1.7b-c2.jsonl $O/r2-1.7b-c2.jsonl $R1/ft-1.7b-free.jsonl $O/r2-1.7b-free.jsonl
touch $O/ALL.done
