#!/bin/sh
# Run train.py on the 4-bit-expert model under a preemptible dibs GPU hold: when another session
# queues for the GPU, train.py saves its state and exits 75, dibs hands over, then reruns it.
#   ./run_train.sh NAME [train.py flags]          output: runs/NAME/ (log: runs/NAME/out.log)
set -u
cd "$(dirname "$0")"
NAME=$1; shift
MODEL=${MODEL:-/home/cyrus/.cache/model-experiments/nkc-train/Qwen3.6-35B-A3B-q4experts}
mkdir -p "runs/$NAME"
exec dibs run gpu --label "npu-kernel-coder LoRA SFT $NAME (preemptible)" --eta "${ETA:-60m}" --preemptible --requeue -- sh -c "
  ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
  K=\$!
  trap 'kill \$K 2>/dev/null' EXIT
  PYTHONPATH=.fla .venv/bin/python train.py '$MODEL' data/train.jsonl data/val.jsonl --out 'runs/$NAME' $*
" >> "runs/$NAME/out.log" 2>&1
