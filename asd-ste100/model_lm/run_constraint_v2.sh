#!/usr/bin/env bash
# Rewrite the 679 doc dev pairs with both fine-tuned models under constraint v2 (generate.py --constraint v2:
# protected-span words copyable, checker-rejected STE words dropped, rejected phrases blocked), then score
# them beside the v1 runs. Waits for the fine-tune run (lm-ft-v1/ALL.done). Detached launch:
#   setsid nohup model_lm/run_constraint_v2.sh > data/out/lm-ft-v1/run-v2.log 2>&1 < /dev/null & disown
# Finished when data/out/lm-ft-v1/V2.done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python
export PYTHONPATH=$PWD/data/out/lm-overlay
VPY=data/out/.venv/bin/python
O=data/out/lm-ft-v1
until [ -e $O/ALL.done ]; do sleep 60; done
gen() {  # name model
  [ -e $O/$1.done ] && return 0
  dibs run gpu --label "asd-ste100 constraint v2 $1" --eta 40m -- $PY model_lm/generate.py --model $2 --out $O/$1.jsonl \
    --shots 0 --constrained --constraint v2
}
gen ft-350m-c2 $O/lfm2.5-350m/best
gen ft-1.7b-c2 $O/lfm3-1.7b/best
env -u PYTHONPATH $VPY model_lm/score.py $O/scores-v2.json $O/ft-350m-c.jsonl $O/ft-350m-c2.jsonl $O/ft-350m-free.jsonl \
  $O/ft-1.7b-c.jsonl $O/ft-1.7b-c2.jsonl $O/ft-1.7b-free.jsonl
touch $O/V2.done
