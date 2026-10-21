#!/bin/bash
# Qwen3.5-9B -> 4B-shaped build (projection with healing, tail polish; the fp32 identity test does not fit 80 GB) on one 80 GB GPU (2026-10-20).
set -uo pipefail
W=/workspace
LOG=$W/ladder.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; exit 1; }
done_flag(){ [ -f "$W/flags/$1.done" ]; }
mark(){ touch "$W/flags/$1.done"; }
mkdir -p "$W/flags" "$W/logs"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MC=$W/model-experiments/manifold-compression
SES="--model Qwen/Qwen3.5-9B --n-fit 64 --n-eval 4 --n-eval-copy 16 --rich --stride 8 --bs 2"

if ! done_flag env; then
  log "=== env ==="
  pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention \
    > "$W/logs/env.log" 2>&1 || fail env
  mark env
fi
cd "$MC"
if ! done_flag build; then
  log "=== build 9B -> 4B shape ==="
  python -u build_projected.py $SES --shape-from Qwen/Qwen3.5-4B --out "$W/proj-4b" --tag _9b > "$W/logs/build.log" 2>&1 || fail build
  log "$(grep 'STUDENT vs teacher' $W/logs/build.log)"
  mark build
fi
if ! done_flag polish; then
  log "=== polish tail ==="
  python -u polish_tail.py $SES --student "$W/proj-4b" --out "$W/proj-4b-tail" --tag _9b_tail4 > "$W/logs/polish.log" 2>&1 || fail polish
  log "$(grep 'POLISHED vs' $W/logs/polish.log)"
  mark polish
fi
tar cf "$W/ladder-out.tar" -C "$W" proj-4b proj-4b-tail -C "$MC" results
log "LADDER DONE"
