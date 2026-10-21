#!/bin/bash
# Shared helpers for the STE pod sessions. Source this; do not run it.
# Required env: RATE = the pod's price in $/hour (read it off the pod page).
# Optional env: BUDGET (default 5), BRANCH, W (workdir), AUTOSTOP=1, HF_TOKEN + OUT_REPO.
set -uo pipefail
: "${RATE:?set RATE to the pod price in dollars per hour, for example RATE=0.79}"
BUDGET=${BUDGET:-5}
W=${W:-/workspace}
BRANCH=${BRANCH:-main}
REPO_URL=https://github.com/Cyronius/model-experiments
LLAMA_TAG=b10673                      # same tag bench/pod_bench.sh uses
TEACHER=${TEACHER:-Qwen/Qwen3.6-35B-A3B-FP8}  # volume and hard-case teacher, served by vLLM
JUDGE_REPO=ggml-org/gpt-oss-20b-GGUF  # judge, a different lineage, served by llama.cpp
SLOTS=${SLOTS:-16}                    # llama.cpp judge slots
VSLOTS=${VSLOTS:-64}                  # vLLM concurrent requests
LOG=$W/run.log
OUT=$W/ste-out
mkdir -p "$W/flags" "$OUT"
[ -f "$W/t0" ] || date +%s > "$W/t0"        # survives a rerun, so cost keeps adding up
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=$W/hf                  # the volume disk, not the container disk
export PATH=/usr/local/cuda/bin:$PATH

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
done_flag(){ [ -f "$W/flags/$1.done" ]; }
mark(){ touch "$W/flags/$1.done"; }
fail(){ log "FAILED in phase $1"; echo "$1" > "$W/FAILED.flag"; pack; stop_pod; exit 1; }

elapsed(){ echo $(( $(date +%s) - $(cat "$W/t0") )); }
spent(){ python3 -c "print(f'{$(elapsed)/3600*$RATE:.2f}')"; }
# seconds left before 85 percent of the budget is gone
remaining(){ python3 -c "print(max(0,int(($BUDGET*0.85/$RATE*3600)-$(elapsed))))"; }
guard(){   # call before each phase
  local left; left=$(remaining)
  log "budget check before $1: spent \$$(spent) of \$$BUDGET, ${left}s left"
  if [ "$left" -le 60 ]; then
    log "BUDGET STOP before $1"; echo "$1" > "$W/STOPPED.flag"; pack; stop_pod; exit 2
  fi
}

pack(){
  tar czf "$W/ste-out.tgz" -C "$W" ste-out run.log 2>/dev/null && log "packed $(du -h "$W/ste-out.tgz" | cut -f1) -> $W/ste-out.tgz"
  if [ -n "${HF_TOKEN:-}" ] && [ -n "${OUT_REPO:-}" ]; then
    hf upload "$OUT_REPO" "$W/ste-out.tgz" ste-out.tgz --repo-type dataset --private >/dev/null 2>&1 && log "uploaded to $OUT_REPO" || log "WARN upload failed; copy $W/ste-out.tgz by hand"
  fi
}
stop_pod(){
  log "RUN ENDED. Spent about \$$(spent). STOP OR TERMINATE THE POD NOW if it is still running."
  if [ "${AUTOSTOP:-0}" = 1 ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
  fi
}

phase_env(){
  done_flag env && return 0
  log "=== env ==="
  pip install -q --break-system-packages -U "huggingface_hub[cli,hf_transfer]" datasets 2>&1 | tail -1
  apt-get install -y -qq cmake ccache psmisc > /dev/null 2>&1 || true
  [ -d "$W/repo" ] || git clone -q --depth 1 --branch "$BRANCH" "$REPO_URL" "$W/repo" || fail env
  python3 -I "$W/repo/asd-ste100/lexicon/build_lexicon.py" --fetch | tee -a "$LOG" || fail env
  [ -d "$W/llama.cpp" ] || git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$W/llama.cpp" || fail env
  cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 || fail env
  cmake --build "$W/llama.cpp/build" -j "$(nproc)" --target llama-server 2>&1 | tail -1 || fail env
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"
  mark env
}

# fetch_model <flag> <repo> <glob> -> sets MODEL_PATH
fetch_model(){
  mkdir -p "$W/models/$1"
  if ! done_flag "model-$1"; then
    log "=== download $2 ($3) ==="
    hf download "$2" --include "$3" --local-dir "$W/models/$1" 2>&1 | tail -1
    mark "model-$1"
  fi
  MODEL_PATH=$(find "$W/models/$1" -name '*.gguf' | sort | head -1)
  [ -n "$MODEL_PATH" ] || { log "no .gguf matched '$3' in $2; fix the glob in common.sh"; fail "model-$1"; }
  log "model file: $MODEL_PATH"
}

phase_vllm(){
  done_flag vllm && return 0
  log "=== vllm install ==="
  pip install -q --break-system-packages uv 2>&1 | tail -1
  [ -d "$W/venv" ] || uv venv -q "$W/venv" || return 1
  uv pip install -q --python "$W/venv/bin/python" -U vllm ninja --torch-backend=auto 2>&1 | tail -2
  "$W/venv/bin/vllm" --version 2>&1 | tail -1 | tee -a "$LOG" || return 1
  mark vllm
}

start_vllm(){   # start_vllm <hf repo> <port>
  # vLLM compiles some kernels at startup and needs ninja and the CUDA toolkit for that.
  export PATH="$W/venv/bin:$PATH" CUDA_HOME=/usr/local/cuda
  local extra=""
  "$W/venv/bin/vllm" serve --help=all 2>/dev/null | grep -q -- "--language-model-only" && extra="--language-model-only"
  nohup "$W/venv/bin/vllm" serve "$1" --port "$2" --served-model-name x --max-model-len 4096 \
        --gpu-memory-utilization 0.92 --max-num-seqs "$VSLOTS" --enable-prefix-caching $extra > "$W/vllm-$2.log" 2>&1 &
  echo $! > "$W/vllm.pid"
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$2/health" > /dev/null && { log "vllm up on $2 ($1)"; return 0; }
    kill -0 "$(cat "$W/vllm.pid")" 2>/dev/null || break; sleep 5
  done
  log "vllm did not come up; tail of its log:"; tail -15 "$W/vllm-$2.log" | tee -a "$LOG"; return 1
}
stop_vllm(){ [ -f "$W/vllm.pid" ] && kill "$(cat "$W/vllm.pid")" 2>/dev/null; sleep 15; true; }

start_server(){   # start_server <model> <port>; RunPod proxies listen on 8081, so use a high port
  pkill -x llama-server 2>/dev/null; sleep 2
  nohup "$W/llama.cpp/build/bin/llama-server" -m "$1" --port "$2" -ngl 99 -np "$SLOTS" -c $((SLOTS*3072)) \
        --cont-batching > "$W/server-$2.log" 2>&1 &
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$2/health" | grep -q ok && { log "server up on $2"; return 0; }; sleep 5
  done
  log "server did not come up; tail of its log:"; tail -5 "$W/server-$2.log" | tee -a "$LOG"; return 1
}

gen(){ timeout "$(remaining)" python3 -I "$W/repo/asd-ste100/data/gen.py" "$@"; }
