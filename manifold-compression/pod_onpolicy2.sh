#!/bin/bash
# Distillation from the 2B: mostly full-length teacher answers (forward KL) plus student answers scored on their first 96 tokens; bench the final checkpoint (2026-10-19).
set -uo pipefail
W=/workspace
LOG=$W/onpolicy.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; exit 1; }
export HF_HOME=$W/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MC=$W/model-experiments/manifold-compression
B=$W/model-experiments/bench
SES="--model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"
log "=== distillation, mixed ==="
cd "$MC" && python -u onpolicy_distill.py $SES --student "$W/proj-0.8b-tail" --out "$W/proj-0.8b-mix2" --tag _mix2 \
  --tokens 2e6 --checkpoints 0.5e6,1e6,2e6 --teacher-frac 0.75 --think-frac 0 --teacher-max-new 768 --micro 2 > "$W/logs/distill_mix2.log" 2>&1 || fail distill
NAMES=""
for c in 2M; do
  name=mc-proj-mix2-$c
  python3 "$W/llama.cpp/convert_hf_to_gguf.py" "$W/proj-0.8b-mix2-$c" --outfile "$W/gguf/$name.gguf" --outtype bf16 --no-mtp \
    > "$W/logs/convert-$name.log" 2>&1 || fail "convert-$name"
  NAMES="$NAMES,$name"
done
NAMES=${NAMES#,}
python3 - <<PY
import json
models = {n: {"gguf": f"$W/gguf/{n}.gguf"} for n in "$NAMES".split(",")}
json.dump({"server_bin": "$W/llama.cpp/build/bin/llama-server", "ppl_bin": "$W/llama.cpp/build/bin/llama-perplexity",
           "models": models}, open("$B/registry.json", "w"), indent=1)
PY
log "=== bench $NAMES ==="
cd "$B" && python3 run_bench.py --runid mix2-20261019 --models "$NAMES" --suites tools,gsm8k,mmlu,ifeval,perplexity \
  > "$W/logs/bench_mix2.log" 2>&1 || fail bench
grep -E ": tools=" "$W/logs/bench_mix2.log" | tee -a "$LOG"
tar czf "$W/onpolicy-results.tgz" -C "$B" results -C "$MC" results
log "MIX2 DONE"
