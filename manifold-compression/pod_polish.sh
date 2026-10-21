#!/bin/bash
# Polish the projected 0.8B's last 4 layers on KL to the 2B, then convert and bench it after the main bench (2026-10-19).
set -uo pipefail
W=/workspace
LOG=$W/project.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
export HF_HOME=$W/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MC=$W/model-experiments/manifold-compression
B=$W/model-experiments/bench
log "=== polish tail ==="
cd "$MC" && python -u polish_tail.py --model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8 \
  --student "$W/proj-0.8b" --out "$W/proj-0.8b-tail" --tag _tail4 > "$W/logs/polish.log" 2>&1 || { log "FAILED polish"; exit 1; }
log "$(grep 'POLISHED vs 2B' $W/logs/polish.log)"
while [ ! -f "$W/flags/bench.done" ]; do sleep 30; done
python3 "$W/llama.cpp/convert_hf_to_gguf.py" "$W/proj-0.8b-tail" --outfile "$W/gguf/mc-proj-0.8b-tail.gguf" --outtype bf16 --no-mtp \
  > "$W/logs/convert-tail.log" 2>&1 || { log "FAILED convert tail"; exit 1; }
python3 - <<PY
import json
p = "$B/registry.json"; r = json.load(open(p))
r["models"]["mc-proj-0.8b-tail"] = {"gguf": "$W/gguf/mc-proj-0.8b-tail.gguf"}
json.dump(r, open(p, "w"), indent=1)
PY
log "=== bench tail ==="
cd "$B" && python3 run_bench.py --runid proj-20261019 --models mc-proj-0.8b-tail --suites tools,gsm8k,mmlu,ifeval,mtbench,perplexity \
  > "$W/logs/bench_tail.log" 2>&1 || { log "FAILED bench tail"; exit 1; }
grep -E ": tools=" "$W/logs/bench_tail.log" | tee -a "$LOG"
tar czf "$W/project-results.tgz" -C "$B" results -C "$MC" results
log "POLISH DONE"
