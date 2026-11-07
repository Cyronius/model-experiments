#!/bin/bash
# Gate 2 CPU re-check (2026-11-07). The local GPU has computed wrong values silently since 2026-11-05
# (memory: local-gpu-silent-faults; the GPU bench of the stock 0.8B had 391 rows of NaN log-probs that
# CPU runs of the same rows do not have). Recomputes on CPU, bf16: the held-out KL of every arm, then the
# multiple-choice bench (MMLU, ARC-C, ARC-E, HellaSwag). Skips steps whose output exists.
cd /home/cyrus/code/model-experiments/manifold-compression
R=results/trunk-gate2-20261105
PY=../target/manifold-rocm/env/bin/python
step() { name=$1; out=$2; shift 2; if [ -f "$out" ]; then echo "$(date +%T) SKIP $name"; return; fi
  echo "$(date +%T) START $name"
  if HIP_VISIBLE_DEVICES= $PY -u trunk_gate2.py "$@" --device cpu --threads 16 > logs/trunk_gate2_$name.cpu.log 2>&1; then echo "$(date +%T) OK $name"; else echo "$(date +%T) FAIL $name"; fi; }
steps() {
  for t in Qwen3.5-0.8B Qwen3.5-2B; do for a in stock fold random; do
    step heldout_${t#Qwen3.5-}_$a $R/$t/$a/held_out_cpu.json heldout --trunk $t --arm $a; done; done
  for t in Qwen3.5-0.8B Qwen3.5-2B; do for a in stock fold random; do
    step bench_${t#Qwen3.5-}_$a $R/$t/$a/bench.json bench --trunk $t --arm $a; done; done
}
export -f step steps; export R PY
dibs run cpu --label "trunk gate2 CPU re-check (held-out KL + bench)" --eta 6h -- bash -c steps
touch $R/cpu_recheck.done
