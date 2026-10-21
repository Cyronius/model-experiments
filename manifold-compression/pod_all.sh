#!/bin/bash
# 2B warp test with every site compressed at once (2026-10-18).
set -uo pipefail
W=/workspace
cd "$W/model-experiments/manifold-compression"
LOG=$W/all.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$W/logs"
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$W/hf
M=Qwen/Qwen3.5-2B-Base
SES="--model $M --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"
log "=== env ==="
pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention > "$W/logs/env.log" 2>&1 || { log FAILED env; exit 1; }
log "=== collect ==="
python -u collect.py $SES > "$W/logs/collect.log" 2>&1 || { log FAILED collect; exit 1; }
log "=== all sites ==="
python -u m_autoencoder_all.py $SES --ks 512,128 --steps 3000 --hidden 2048 > "$W/logs/all.log" 2>&1 || { log FAILED all; exit 1; }
log "ALL DONE"
