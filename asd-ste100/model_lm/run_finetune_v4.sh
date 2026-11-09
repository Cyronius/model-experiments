#!/usr/bin/env bash
# Fine-tune round 4, LFM2.5-350M: round 3's recipe (3 epochs, LR 3e-5, identity pairs, epoch by held-out loss)
# on data/out/lm-doc-v3 (model_lm/doc_pairs_v3.py: lm-doc-v2 plus 1,990 Claude subagent pairs on doc pool v6
# and 2,895 real checker-clean pool-v6 passages as identity rows). Dev rewrites with constraint v2 and free,
# a 24-row CPU canary, scores beside rounds 1-3. Detached launch:
#   setsid nohup model_lm/run_finetune_v4.sh > data/out/lm-ft-v4/run.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-ft-v4/ALL.done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
export PYTHONPATH=$PWD/data/out/lm-overlay
VPY=data/out/.venv/bin/python
O=data/out/lm-ft-v4
R3=data/out/lm-ft-v3
R1=data/out/lm-ft-v1
R2=data/out/lm-ft-v2
mkdir -p $O
for try in 1 2 3; do  # a non-finite loss from a GPU fault stops finetune.py, which resumes after the last saved epoch
  [ -e $O/lfm2.5-350m/FT.done ] && break
  dibs run gpu --label "asd-ste100 LM fine-tune round 4 350m" --eta 140m -- $PY model_lm/finetune.py \
    --model data/out/lfm/LFM2.5-350M --out $O/lfm2.5-350m --lr 3e-5 --epochs 3 --identity --data data/out/lm-doc-v3 \
    || echo "fine-tune attempt $try failed" >&2
done
[ -e $O/lfm2.5-350m/FT.done ]
gen() {  # name flags...
  local name=$1; shift
  [ -e $O/$name.done ] && return 0
  dibs run gpu --label "asd-ste100 LM round 4 eval $name" --eta 15m -- $PY model_lm/generate.py --model $O/lfm2.5-350m/best \
    --out $O/$name.jsonl --shots 0 "$@"
}
gen r4-350m-c2 --constrained --constraint v2
gen r4-350m-free
[ -e $O/canary.done ] || dibs run cpu --label "asd-ste100 round 4 CPU canary" --eta 15m -- $PY model_lm/generate.py \
  --model $O/lfm2.5-350m/best --out $O/canary.jsonl --shots 0 --constrained --constraint v2 --limit 24 --inputs src \
  --device cpu --batch 8
$PY - <<'PYEOF'
import json
O = 'data/out/lm-ft-v4'
g = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/r4-350m-c2.jsonl')) if j['input'] == 'src'}
c = {j['id']: j['output'] for j in map(json.loads, open(f'{O}/canary.jsonl'))}
diff = [i for i in c if g.get(i) != c[i]]
json.dump(dict(rows=len(c), differ=len(diff), ids=diff), open(f'{O}/canary.json', 'w'), indent=2)
print('canary: %d of %d CPU rows differ from the GPU run' % (len(diff), len(c)))
PYEOF
env -u PYTHONPATH $VPY model_lm/score.py $O/scores.json $R1/ft-350m-c2.jsonl $R2/r2-350m-c2.jsonl $R3/r3-350m-c2.jsonl $O/r4-350m-c2.jsonl \
  $R1/ft-350m-free.jsonl $R2/r2-350m-free.jsonl $R3/r3-350m-free.jsonl $O/r4-350m-free.jsonl
touch $O/ALL.done
