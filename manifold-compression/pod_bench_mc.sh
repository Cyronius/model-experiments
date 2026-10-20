#!/bin/bash
# Bench the MLP-shrunk Qwen3.5-2B chat builds against stock 2B and stock 0.8B with the repo's bench harness on llama.cpp (2026-10-18).
set -uo pipefail
W=/workspace
LOG=$W/bench.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; echo "$1" > "$W/BENCH_FAILED.flag"; exit 1; }
done_flag(){ [ -f "$W/flags/b-$1.done" ]; }
mark(){ touch "$W/flags/b-$1.done"; }
mkdir -p "$W/flags" "$W/gguf"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PATH=/usr/local/cuda/bin:$PATH

if ! done_flag llama; then
  log "=== llama.cpp CUDA build ==="
  apt-get install -y -qq cmake > /dev/null 2>&1 || true
  [ -d "$W/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$W/llama.cpp" || fail clone
  cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release > "$W/logs/llama-cmake.log" 2>&1 || fail cmake
  cmake --build "$W/llama.cpp/build" -j "$(nproc)" --target llama-server llama-perplexity > "$W/logs/llama-build.log" 2>&1 || fail build
  pip install -q --break-system-packages "$W/llama.cpp/gguf-py" sentencepiece > /dev/null 2>&1 || fail gguf-py
  mark llama
fi

while [ ! -f "$W/BUILT.flag" ]; do sleep 20; done

declare -A SRC
SRC[stock-2b]=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-2B'))")
SRC[stock-0.8b]=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-0.8B'))")
SRC[mc-2b-mlp75]=$W/pruned-2b-chat-mlp75
SRC[mc-2b-mlp50]=$W/pruned-2b-chat-mlp50
for name in stock-2b stock-0.8b mc-2b-mlp75 mc-2b-mlp50; do
  done_flag "conv-$name" && continue
  log "=== convert $name ==="
  python3 "$W/llama.cpp/convert_hf_to_gguf.py" "${SRC[$name]}" --outfile "$W/gguf/$name.gguf" --outtype bf16 \
    > "$W/logs/convert-$name.log" 2>&1 || fail "convert-$name"
  mark "conv-$name"
done

B=$W/model-experiments/bench
python3 - <<PY
import json
models = {n: {"gguf": f"$W/gguf/{n}.gguf"} for n in ["stock-2b", "stock-0.8b", "mc-2b-mlp75", "mc-2b-mlp50"]}
json.dump({"server_bin": "$W/llama.cpp/build/bin/llama-server", "ppl_bin": "$W/llama.cpp/build/bin/llama-perplexity",
           "models": models}, open("$B/registry.json", "w"), indent=1)
PY
if ! done_flag bench; then
  log "=== bench ==="
  cd "$B" && python3 run_bench.py --runid mc-pod-20261018 --models stock-2b,stock-0.8b,mc-2b-mlp75,mc-2b-mlp50 \
    --suites tools,gsm8k,mmlu,perplexity 2>&1 | tee -a "$LOG" || fail bench
  mark bench
fi
tar czf "$W/bench-results.tgz" -C "$B" results
log "BENCH DONE"
touch "$W/BENCH_DONE.flag"
