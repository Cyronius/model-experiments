#!/bin/bash
# Gate 2 bench (multiple-choice panel), detached. The owner asked for a 5-hour pause first (2026-11-06 20:30 CST).
cd /home/cyrus/code/model-experiments/manifold-compression
R=results/trunk-gate2-20261105
echo "$(date +%T) sleeping 5 h"
sleep 18000
PY=../target/manifold-rocm/env/bin/python
step() { name=$1; out=$2; shift 2; if [ -f "$out" ]; then echo "$(date +%T) SKIP $name"; return; fi
  echo "$(date +%T) START $name"
  if $PY -u trunk_gate2.py "$@" --device cuda > logs/trunk_gate2_$name.log 2>&1; then echo "$(date +%T) OK $name"; else echo "$(date +%T) FAIL $name"; fi; }
steps() {
for t in Qwen3.5-0.8B Qwen3.5-2B; do for a in stock fold random; do
  step bench_${t#Qwen3.5-}_$a $R/$t/$a/bench.json bench --trunk $t --arm $a; done; done
step bench_4B_stock $R/Qwen3.5-4B/stock/bench.json bench --trunk Qwen3.5-4B --arm stock
}
export -f step steps; export R PY
dibs run gpu --label "trunk gate2 bench: MMLU/ARC/HellaSwag, 7 arms" --eta 2h -- bash -c steps
touch $R/bench.done
