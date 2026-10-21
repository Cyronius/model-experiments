#!/bin/bash
# Rebuild the 50% chat model two ways (own-chat calibration; plus sensitivity weighting), then convert and bench both (2026-10-18).
set -uo pipefail
W=/workspace
cd "$W/model-experiments/manifold-compression"
LOG=$W/shape.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
SES="--model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8 --chat --keep 0.5"
log "=== build selfchat ==="
python -u build_pruned.py $SES --self-chat 192 --tag _selfchat --out "$W/pruned-2b-chat-mlp50-selfchat" > "$W/logs/build50_selfchat.log" 2>&1 || { log FAILED build selfchat; exit 1; }
log "=== build selfchat+shape ==="
python -u build_pruned.py $SES --self-chat 192 --shape 0.5 --tag _shaped --out "$W/pruned-2b-chat-mlp50-shaped" > "$W/logs/build50_shaped.log" 2>&1 || { log FAILED build shaped; exit 1; }
for name in selfchat shaped; do
  log "=== convert $name ==="
  python3 "$W/llama.cpp/convert_hf_to_gguf.py" "$W/pruned-2b-chat-mlp50-$name" --outfile "$W/gguf/mc-2b-mlp50-$name.gguf" --outtype bf16 \
    > "$W/logs/convert-$name.log" 2>&1 || { log FAILED convert $name; exit 1; }
done
B=$W/model-experiments/bench
python3 - <<PY
import json
p = "$B/registry.json"
r = json.load(open(p))
for n in ["mc-2b-mlp50-selfchat", "mc-2b-mlp50-shaped"]:
    r["models"][n] = {"gguf": f"$W/gguf/{n}.gguf"}
json.dump(r, open(p, "w"), indent=1)
PY
log "=== bench ==="
cd "$B" && python3 run_bench.py --runid mc-pod-20261018 --models mc-2b-mlp50-selfchat,mc-2b-mlp50-shaped \
  --suites tools,gsm8k,mmlu,perplexity 2>&1 | tee -a "$LOG" || { log FAILED bench; exit 1; }
log "SHAPE DONE"
