#!/bin/bash
# Rebuild the projected 0.8B, polish its tail, run on-policy distillation from the 2B, then bench the checkpoints (2026-10-19).
set -uo pipefail
W=/workspace
LOG=$W/onpolicy.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; exit 1; }
done_flag(){ [ -f "$W/flags/$1.done" ]; }
mark(){ touch "$W/flags/$1.done"; }
mkdir -p "$W/flags" "$W/logs" "$W/gguf"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/usr/local/cuda/bin:$PATH
MC=$W/model-experiments/manifold-compression
B=$W/model-experiments/bench
SES="--model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"

if ! done_flag env; then
  log "=== env ==="
  pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention \
    lm_eval langdetect immutabledict nltk sentencepiece > "$W/logs/env.log" 2>&1 || fail env
  mark env
fi
if ! done_flag llama; then
  ( apt-get install -y -qq cmake > /dev/null 2>&1
    [ -d "$W/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$W/llama.cpp"
    cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF \
          -DCMAKE_BUILD_TYPE=Release > "$W/logs/llama-cmake.log" 2>&1 &&
    cmake --build "$W/llama.cpp/build" -j 16 --target llama-server llama-perplexity > "$W/logs/llama-build.log" 2>&1 &&
    pip install -q --break-system-packages "$W/llama.cpp/gguf-py" > /dev/null 2>&1 && touch "$W/flags/llama.done" ) &
fi
cd "$MC"
if ! done_flag build; then
  log "=== rebuild projected 0.8B ==="
  python -u build_projected.py $SES --out "$W/proj-0.8b" --tag _e2 > "$W/logs/build.log" 2>&1 || fail build
  mark build
fi
if ! done_flag polish; then
  log "=== polish tail ==="
  python -u polish_tail.py $SES --student "$W/proj-0.8b" --out "$W/proj-0.8b-tail" --tag _tail4b > "$W/logs/polish.log" 2>&1 || fail polish
  log "$(grep 'POLISHED vs 2B' $W/logs/polish.log)"
  mark polish
fi
if ! done_flag distill; then
  log "=== on-policy distillation ==="
  python -u onpolicy_distill.py $SES --student "$W/proj-0.8b-tail" --out "$W/proj-0.8b-op" --tag _a > "$W/logs/distill.log" 2>&1 || fail distill
  mark distill
fi
while [ ! -f "$W/flags/llama.done" ]; do sleep 20; done
NAMES=""
for d in "$W"/proj-0.8b-op-*M; do
  name=mc-proj-op-$(basename "$d" | sed 's/proj-0.8b-op-//')
  done_flag "conv-$name" || { python3 "$W/llama.cpp/convert_hf_to_gguf.py" "$d" --outfile "$W/gguf/$name.gguf" --outtype bf16 --no-mtp \
    > "$W/logs/convert-$name.log" 2>&1 || fail "convert-$name"; mark "conv-$name"; }
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
cd "$B" && python3 run_bench.py --runid op-20261019 --models "$NAMES" --suites tools,gsm8k,mmlu,ifeval,mtbench,perplexity \
  > "$W/logs/bench.log" 2>&1 || fail bench
grep -E ": tools=" "$W/logs/bench.log" | tee -a "$LOG"
tar czf "$W/onpolicy-results.tgz" -C "$B" results -C "$MC" results
log "ONPOLICY DONE"
