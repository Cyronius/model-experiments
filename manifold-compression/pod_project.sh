#!/bin/bash
# 2B -> 0.8B-shaped build with layer-wise healing, then convert and bench it against stock 0.8B and stock 2B (2026-10-19).
set -uo pipefail
W=/workspace
LOG=$W/project.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; exit 1; }
done_flag(){ [ -f "$W/flags/$1.done" ]; }
mark(){ touch "$W/flags/$1.done"; }
mkdir -p "$W/flags" "$W/logs" "$W/gguf"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/usr/local/cuda/bin:$PATH
MC=$W/model-experiments/manifold-compression
B=$W/model-experiments/bench

if ! done_flag env; then
  log "=== env ==="
  pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention \
    lm_eval langdetect immutabledict nltk sentencepiece > "$W/logs/env.log" 2>&1 || fail env
  mark env
fi

# llama.cpp builds on the CPU while the GPU does the projection; -j 16 keeps it under the pod's RAM cap.
if ! done_flag llama; then
  ( apt-get install -y -qq cmake > /dev/null 2>&1
    [ -d "$W/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$W/llama.cpp"
    cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF \
          -DCMAKE_BUILD_TYPE=Release > "$W/logs/llama-cmake.log" 2>&1 &&
    cmake --build "$W/llama.cpp/build" -j 16 --target llama-server llama-perplexity > "$W/logs/llama-build.log" 2>&1 &&
    pip install -q --break-system-packages "$W/llama.cpp/gguf-py" > /dev/null 2>&1 && touch "$W/flags/llama.done" ) &
fi

if ! done_flag build; then
  log "=== build 2B -> 0.8B ==="
  cd "$MC" && python -u build_projected.py --model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8 \
    --out "$W/proj-0.8b" --tag _e > "$W/logs/build.log" 2>&1 || fail build
  mark build
fi
log "$(grep 'STUDENT vs 2B' $W/logs/build.log)"

while [ ! -f "$W/flags/llama.done" ]; do sleep 20; done
declare -A SRC
SRC[stock-2b]=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-2B'))")
SRC[stock-0.8b]=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-0.8B'))")
SRC[mc-proj-0.8b]=$W/proj-0.8b
for name in mc-proj-0.8b stock-0.8b stock-2b; do
  done_flag "conv-$name" && continue
  log "=== convert $name ==="
  python3 "$W/llama.cpp/convert_hf_to_gguf.py" "${SRC[$name]}" --outfile "$W/gguf/$name.gguf" --outtype bf16 --no-mtp \
    > "$W/logs/convert-$name.log" 2>&1 || fail "convert-$name"
  mark "conv-$name"
done

python3 - <<PY
import json
models = {n: {"gguf": f"$W/gguf/{n}.gguf"} for n in ["mc-proj-0.8b", "stock-0.8b", "stock-2b"]}
json.dump({"server_bin": "$W/llama.cpp/build/bin/llama-server", "ppl_bin": "$W/llama.cpp/build/bin/llama-perplexity",
           "models": models}, open("$B/registry.json", "w"), indent=1)
PY
if ! done_flag bench; then
  log "=== bench ==="
  cd "$B" && python3 run_bench.py --runid proj-20261019 --models mc-proj-0.8b,stock-0.8b,stock-2b \
    --suites tools,gsm8k,mmlu,ifeval,mtbench,perplexity > "$W/logs/bench.log" 2>&1 || fail bench
  grep -E ": tools=" "$W/logs/bench.log" | tee -a "$LOG"
  mark bench
fi
tar czf "$W/project-results.tgz" -C "$B" results -C "$MC" results
log "PROJECT DONE"
