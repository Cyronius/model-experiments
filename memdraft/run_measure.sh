#!/bin/bash
# The clean timing: step costs and drafter arms for one model, each inside `dibs run --measure`.
# --measure takes the whole machine and holds other sessions' dibs requests while it waits for
# quiet, so start this by hand at a time when that is acceptable. It is not started automatically.
#
#   memdraft/run_measure.sh 9b vk        # Qwen3.5-9B on the GPU (Vulkan)
#   memdraft/run_measure.sh 2b cpu       # Qwen3.5-2B on 8 CPU threads
set -eu
cd "$(dirname "$0")/.."
MODEL=${1:?9b or 2b}; BACKEND=${2:?vk or cpu}
PY=target/manifold-audit/env/bin/python
S=$HOME/.cache/model-experiments/memdraft
T=$PWD/target/memdraft/9b-all-c3.mdt
case $MODEL in 9b) GGUF=$S/models/Qwen3.5-9B-Q4_K_M.gguf;; 2b) GGUF=$S/models/Qwen3.5-2B-Q4_K_M.gguf;; *) echo "unknown model $MODEL"; exit 2;; esac
case $BACKEND in
  vk)  SERVER=$S/llama.cpp/build-vk/bin/llama-server;  RES="gpu port-8455"; GUARD="--gpu-guard"; COMMON="-ngl 99 -c 8192";;
  cpu) SERVER=$S/llama.cpp/build-cpu/bin/llama-server; RES="port-8455";     GUARD="";            COMMON="-ngl 0 -t 8 -c 8192";;
  *) echo "unknown backend $BACKEND"; exit 2;;
esac
STAMP=$(date +%Y%m%d-%H%M)
TABLE="--spec-type ngram-table --spec-table-file $T"
bench() {  # name eta bench-options... -- arms...
  local name=$1 eta=$2; shift 2
  dibs run --measure $RES --label "MemDraft $name $MODEL $BACKEND" --eta "$eta" -- \
    $PY memdraft/live_bench.py --clean $GUARD --server "$SERVER" --model "$GGUF" --prompts memdraft/prompts/live-v1.jsonl \
    --out "memdraft/results/$name-$MODEL-$BACKEND-$STAMP.json" "$@" 2>&1 | tee "target/memdraft/$name-$MODEL-$BACKEND-$STAMP.log"
}
const() { echo "--spec-type ngram-table --spec-table-const $1 --spec-table-n-max $1 --spec-synth-rates $2"; }

# What a step costs: n draft tokens all accepted, all rejected, and drafting only every other step
# (which changes the batch shape at every step).
bench cost 15m --per-domain 1 --max-tokens 192 --rounds 2 -- $COMMON \
  --arm none= \
  --arm "accept1=$(const 1 1)" --arm "accept2=$(const 2 1,1)" --arm "accept4=$(const 4 1,1,1,1)" \
  --arm "reject1=$(const 1 0)" --arm "reject2=$(const 2 0,0)" --arm "reject4=$(const 4 0,0,0,0)" \
  --arm "reject2-every-other-step=$(const 2 0,0) --spec-table-const-period 2" \
  --arm "reject2-checkpoints=$(const 2 0,0) --spec-table-no-rs"

# The drafter, against llama.cpp's own free drafters.
bench draft 30m --per-domain 3 --max-tokens 160 -- $COMMON \
  --arm none= \
  --arm "both-0.15=$TABLE --spec-table-tau 0.15" --arm "both-0.3=$TABLE --spec-table-tau 0.3" --arm "both-0.5=$TABLE --spec-table-tau 0.5" \
  --arm "table-0.3=$TABLE --spec-table-tau 0.3 --spec-table-source table" \
  --arm "context-0.3=$TABLE --spec-table-tau 0.3 --spec-table-source context" \
  --arm "pad1=$TABLE --spec-table-n-max 1 --spec-table-pad" --arm "pad2=$TABLE --spec-table-n-max 2 --spec-table-pad" \
  --arm "pad3=$TABLE --spec-table-n-max 3 --spec-table-pad" \
  --arm "ngram-simple=--spec-type ngram-simple" --arm "ngram-mod=--spec-type ngram-mod" --arm "ngram-cache=--spec-type ngram-cache"
