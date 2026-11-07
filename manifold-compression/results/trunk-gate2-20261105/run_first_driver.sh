#!/bin/bash
# Gate 2, everything after the 0.8B fold run. Each step logs to logs/trunk_gate2_<name>.log; a failure does not stop independent steps.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
run() { name=$1; shift; echo "$(date +%T) START $name"; if $PY -u trunk_gate2.py "$@" --device cuda > logs/trunk_gate2_$name.log 2>&1; then echo "$(date +%T) OK $name"; else echo "$(date +%T) FAIL $name"; fi; }
run train08_random train --trunk Qwen3.5-0.8B --arm random --steps-from fold
run check08_stock_off check --trunk Qwen3.5-0.8B --arm stock --thinking off
run check08_fold_off check --trunk Qwen3.5-0.8B --arm fold --thinking off
run check08_random_off check --trunk Qwen3.5-0.8B --arm random --thinking off
run train2b_fold train --trunk Qwen3.5-2B --arm fold --hours 1.5
run train2b_random train --trunk Qwen3.5-2B --arm random --steps-from fold
run check2b_fold_off check --trunk Qwen3.5-2B --arm fold --thinking off
run check2b_random_off check --trunk Qwen3.5-2B --arm random --thinking off
run check08_stock_on check --trunk Qwen3.5-0.8B --arm stock --thinking on
run check08_fold_on check --trunk Qwen3.5-0.8B --arm fold --thinking on
run check08_random_on check --trunk Qwen3.5-0.8B --arm random --thinking on
run check2b_fold_on check --trunk Qwen3.5-2B --arm fold --thinking on
run check2b_random_on check --trunk Qwen3.5-2B --arm random --thinking on
echo "$(date +%T) ALL DONE"
