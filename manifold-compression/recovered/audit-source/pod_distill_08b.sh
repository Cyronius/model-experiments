#!/bin/bash
# 2B -> 0.8B-shaped build, tail polish, then whole-network distillation with checkpoints (2026-10-21).
set -uo pipefail
W=/workspace
LOG=$W/distill.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FAILED in $1"; exit 1; }
done_flag(){ [ -f "$W/flags/$1.done" ]; }
mark(){ touch "$W/flags/$1.done"; }
mkdir -p "$W/flags" "$W/logs"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MC=$W/model-experiments/manifold-compression
SES="--model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"
cd "$MC"

if ! done_flag env; then
  log "=== env ==="
  pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention \
    > "$W/logs/env.log" 2>&1 || fail env
  mark env
fi
if ! done_flag build; then
  log "=== build 2B -> 0.8B shape ==="
  python -u build_projected.py $SES --out "$W/proj-0.8b" --tag _d > "$W/logs/build.log" 2>&1 || fail build
  log "$(grep 'STUDENT vs teacher' $W/logs/build.log)"
  mark build
fi
if ! done_flag polish; then
  log "=== polish tail ==="
  python -u polish_tail.py $SES --student "$W/proj-0.8b" --out "$W/proj-0.8b-tail" --tag _d_tail4 > "$W/logs/polish.log" 2>&1 || fail polish
  log "$(grep 'POLISHED vs' $W/logs/polish.log)"
  mark polish
fi
log "=== distill ==="
python -u distill_whole.py --student "$W/proj-0.8b-tail" --out "$W/distill" --rows-cache "$W/rows.pt" > "$W/logs/distill.log" 2>&1 || fail distill
log "DISTILL DONE"
