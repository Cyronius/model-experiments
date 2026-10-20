#!/bin/bash
# Phase 1 on a rented GPU: Qwen3.5-2B-Base through collect + M1-M5 + decoder KL; flag-gated so a rerun resumes (2026-10-18).
set -uo pipefail
W=${W:-/workspace}
cd "$W/model-experiments/manifold-compression"
LOG=$W/run.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
step(){ local name=$1; shift
  [ -f "$W/flags/$name.done" ] && { log "skip $name (done)"; return 0; }
  log "=== $name ==="
  "$@" >> "$W/logs/$name.log" 2>&1 || { log "FAILED $name (see logs/$name.log)"; echo "$name" > "$W/FAILED.flag"; exit 1; }
  touch "$W/flags/$name.done"; }
mkdir -p "$W/flags" "$W/logs"
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$W/hf

M=Qwen/Qwen3.5-2B-Base
SES="--model $M --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"

step env pip install -q --break-system-packages -U "transformers==5.14.1" "huggingface_hub[hf_transfer]" pandas pyarrow flash-linear-attention
step check python -c "import torch, fla, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())"
step collect python -u collect.py $SES
step idim python -u m_idim.py --model $M --n-points 4000
step width python -u m_width.py $SES --ks 0.25,0.5,0.75 --site-stride 4 --mlp-steps 1500 --mlp-hidden 2048
step boundary python -u m_boundary.py $SES --mlp-steps 1500 --greedy-candidates 3
step boundary_tail python -u m_boundary.py $SES --mlp-steps 1500 --greedy-candidates 3 --greedy-only --greedy-score mean+copy_p99 --greedy-budget 1.0
step perturb python -u m_perturb.py $SES --site-stride 2 --eps 0.1
step decoder_kl python -u m_decoder_kl.py $SES --sites 12,30,40 --ks 512,1024 --steps 3000
step report python report.py --model $M
log "ALL DONE"
touch "$W/DONE.flag"
