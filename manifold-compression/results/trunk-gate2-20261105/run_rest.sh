#!/bin/bash
# Gate 2 backstop driver (detached from the Claude session). Waits for the first driver (dibs pid given as $1)
# to exit, then runs every step whose output is still missing. If the first driver finished, it does nothing.
cd /home/cyrus/code/model-experiments/manifold-compression
R=results/trunk-gate2-20261105
while kill -0 "$1" 2>/dev/null; do sleep 30; done
PY=../target/manifold-rocm/env/bin/python
step() { name=$1; out=$2; shift 2; if [ -f "$out" ]; then echo "$(date +%T) SKIP $name"; return; fi
  echo "$(date +%T) START $name"
  if $PY -u trunk_gate2.py "$@" --device cuda > logs/trunk_gate2_$name.b.log 2>&1; then echo "$(date +%T) OK $name"; else echo "$(date +%T) FAIL $name"; fi; }
steps() {
step train08_random $R/Qwen3.5-0.8B/random/train.json train --trunk Qwen3.5-0.8B --arm random --steps-from fold
for a in stock fold random; do step check08_${a}_off $R/Qwen3.5-0.8B/$a/check_thinking_off.json check --trunk Qwen3.5-0.8B --arm $a --thinking off; done
step train2b_fold $R/Qwen3.5-2B/fold/train.json train --trunk Qwen3.5-2B --arm fold --hours 1.5
step train2b_random $R/Qwen3.5-2B/random/train.json train --trunk Qwen3.5-2B --arm random --steps-from fold
for a in fold random; do step check2b_${a}_off $R/Qwen3.5-2B/$a/check_thinking_off.json check --trunk Qwen3.5-2B --arm $a --thinking off; done
for a in stock fold random; do step check08_${a}_on $R/Qwen3.5-0.8B/$a/check_thinking_on.json check --trunk Qwen3.5-0.8B --arm $a --thinking on; done
for a in fold random; do step check2b_${a}_on $R/Qwen3.5-2B/$a/check_thinking_on.json check --trunk Qwen3.5-2B --arm $a --thinking on; done
}
if [ -f $R/Qwen3.5-2B/random/check_thinking_on.json ]; then echo "$(date +%T) first driver finished everything"; else
  export -f step steps; export R PY
  dibs run gpu --label "trunk gate2 backstop: missing steps" --eta 6h -- bash -c steps
fi
touch $R/gate2.done
