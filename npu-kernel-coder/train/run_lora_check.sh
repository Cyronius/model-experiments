#!/bin/sh
# Run lora_check.py on the 4-bit-expert model (q4moe.py) under a dibs GPU hold, renewed every 20 min.
#   ./run_lora_check.sh [lora_check.py flags]        log: runs/lora-check.log
set -u
cd "$(dirname "$0")"
MODEL=${MODEL:-/home/cyrus/.cache/model-experiments/nkc-train/Qwen3.6-35B-A3B-q4experts}
mkdir -p runs
exec dibs run gpu --label "npu-kernel-coder LoRA feasibility (Qwen3.6-35B-A3B, 4-bit experts)" --eta "${ETA:-60m}" -- sh -c "
  ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
  K=\$!
  trap 'kill \$K 2>/dev/null' EXIT
  .venv/bin/python lora_check.py '$MODEL' data/train.jsonl data/val.jsonl $*
"
