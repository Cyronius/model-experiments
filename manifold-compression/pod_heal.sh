#!/bin/bash
# Layer-by-layer healing on a rented GPU: PCA arm, then PCA + learned warp arm (2026-10-18).
set -uo pipefail
W=/workspace
cd "$W/model-experiments/manifold-compression"
LOG=$W/heal.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$W/logs"
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$W/hf
M=${M:-Qwen/Qwen3.5-0.8B}
K=${K:-256}
SES="--model $M --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"
log "=== env ==="
pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention > "$W/logs/env.log" 2>&1 || { log FAILED env; exit 1; }
log "=== pca arm ==="
python -u heal_layerwise.py $SES --k $K --tag _pca > "$W/logs/heal_pca.log" 2>&1 || { log FAILED pca; exit 1; }
log "=== warp arm ==="
python -u heal_layerwise.py $SES --k $K --warp --tag _warp > "$W/logs/heal_warp.log" 2>&1 || { log FAILED warp; exit 1; }
log "HEAL DONE"
