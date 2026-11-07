#!/usr/bin/env bash
# Fine-tune LFM2.5-350M and LFM3-1.7B on the 6,897 doc train pairs (model_lm/finetune.py), then rewrite the
# 679 doc dev pairs with each selected epoch, zero-shot prompt, constrained and free, a 24-row CPU canary
# for the 1.7B constrained run, and scores. Waits for the few-shot run (lm-zeroshot-v1/ALL.done) first.
# Detached launch:
#   setsid nohup model_lm/run_finetune.sh > data/out/lm-ft-v1/run.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-ft-v1/ALL.done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
export PYTHONPATH=$PWD/data/out/lm-overlay
VPY=data/out/.venv/bin/python
O=data/out/lm-ft-v1
mkdir -p $O
until [ -e data/out/lm-zeroshot-v1/ALL.done ]; do sleep 60; done
ft() {  # name model lr; retried: a non-finite loss from a GPU fault stops finetune.py, which resumes after the last saved epoch
  local try
  for try in 1 2 3; do
    [ -e $O/$1/FT.done ] && return 0
    dibs run gpu --label "asd-ste100 LM fine-tune $1" --eta 3h -- $PY model_lm/finetune.py --model $2 --out $O/$1 --lr $3 --epochs 3 \
      && return 0
    echo "fine-tune $1 attempt $try failed" >&2
  done
  return 1
}
gen() {  # name model flags...
  local name=$1 model=$2; shift 2
  [ -e $O/$name.done ] && return 0
  dibs run gpu --label "asd-ste100 LM fine-tuned eval $name" --eta 1h -- $PY model_lm/generate.py --model $model --out $O/$name.jsonl --shots 0 "$@"
}
ft lfm2.5-350m data/out/lfm/LFM2.5-350M 3e-5
gen ft-350m-c $O/lfm2.5-350m/best --constrained
gen ft-350m-free $O/lfm2.5-350m/best
ft lfm3-1.7b data/out/lfm/LFM3-1.7B 1e-5
gen ft-1.7b-c $O/lfm3-1.7b/best --constrained
gen ft-1.7b-free $O/lfm3-1.7b/best
if [ ! -e $O/canary-cpu.done ]; then
  dibs run cpu --label "asd-ste100 fine-tuned CPU canary" --eta 40m -- $PY model_lm/generate.py --model $O/lfm3-1.7b/best \
    --out $O/canary-cpu.jsonl --shots 0 --constrained --limit 24 --inputs src --device cpu --batch 8
fi
$PY - <<'PYEOF'
import json
O='data/out/lm-ft-v1'
g={j['id']:j['output'] for j in map(json.loads,open(f'{O}/ft-1.7b-c.jsonl')) if j['input']=='src'}
c={j['id']:j['output'] for j in map(json.loads,open(f'{O}/canary-cpu.jsonl'))}
diff=[i for i in c if g.get(i)!=c[i]]
json.dump(dict(rows=len(c),differ=len(diff),ids=diff),open(f'{O}/canary.json','w'),indent=2)
print('canary: %d of %d CPU rows differ from the GPU run' % (len(diff),len(c)))
PYEOF
env -u PYTHONPATH $VPY model_lm/score.py $O/scores.json $O/ft-350m-c.jsonl $O/ft-350m-free.jsonl $O/ft-1.7b-c.jsonl $O/ft-1.7b-free.jsonl
touch $O/ALL.done
