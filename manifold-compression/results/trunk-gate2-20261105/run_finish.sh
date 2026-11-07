#!/bin/bash
# Gate 2 finish (2026-11-07), detached. The machine hung ~02:05 on 11-07 during bench_0.8B_random and rebooted at 05:58.
# Runs every step whose output is still missing: the stock 0.8B GSM8K checks (they failed on a missing-folder bug, now fixed)
# first, then the multiple-choice bench for the remaining arms.
cd /home/cyrus/code/model-experiments/manifold-compression
R=results/trunk-gate2-20261105
PY=../target/manifold-rocm/env/bin/python
step() { name=$1; out=$2; shift 2; if [ -f "$out" ]; then echo "$(date +%T) SKIP $name"; return; fi
  echo "$(date +%T) START $name"
  if $PY -u trunk_gate2.py "$@" --device cuda > logs/trunk_gate2_$name.f.log 2>&1; then echo "$(date +%T) OK $name"; else echo "$(date +%T) FAIL $name"; fi; }
steps() {
for m in off on; do step check08_stock_$m $R/Qwen3.5-0.8B/stock/check_thinking_$m.json check --trunk Qwen3.5-0.8B --arm stock --thinking $m; done
for t in Qwen3.5-0.8B Qwen3.5-2B; do for a in stock fold random; do
  step bench_${t#Qwen3.5-}_$a $R/$t/$a/bench.json bench --trunk $t --arm $a; done; done
step bench_4B_stock $R/Qwen3.5-4B/stock/bench.json bench --trunk Qwen3.5-4B --arm stock
}
export -f step steps; export R PY
dibs run gpu --label "trunk gate2 finish: stock 0.8B GSM8K + bench" --eta 2h -- bash -c steps
touch $R/finish.done
