#!/bin/sh
# Run train.py on the 4-bit-expert model under a preemptible dibs GPU hold: when another session
# queues for the GPU, train.py saves its state and exits 75, dibs hands over, then reruns it.
# If memguard stops it (exit 76: RAM ran low), wait until MemAvailable is back above
# $RESUME_GB (default 30) for 5 minutes, then queue again; any other exit ends the loop.
#   ./run_train.sh NAME [train.py flags]          output: runs/NAME/ (log: runs/NAME/out.log)
set -u
cd "$(dirname "$0")"
NAME=$1; shift
MODEL=${MODEL:-/home/cyrus/.cache/model-experiments/nkc-train/Qwen3.6-35B-A3B-q4experts}
mkdir -p "runs/$NAME"
avail_gb() { awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
while :; do
  dibs run gpu --label "npu-kernel-coder LoRA SFT $NAME (preemptible)" --eta "${ETA:-60m}" --preemptible --requeue -- sh -c "
    ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
    K=\$!
    trap 'kill \$K 2>/dev/null' EXIT
    PYTHONPATH=.fla .venv/bin/python train.py '$MODEL' data/train.jsonl data/val.jsonl --out 'runs/$NAME' $*
  " >> "runs/$NAME/out.log" 2>&1
  rc=$?
  [ "$rc" = 76 ] || exit "$rc"
  echo "run_train: memguard stop at $(date '+%F %T'); waiting for ${RESUME_GB:-30} GB free" >> "runs/$NAME/out.log"
  ok=0
  while [ "$ok" -lt 60 ]; do
    if [ "$(avail_gb)" -ge "${RESUME_GB:-30}" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 5
  done
done
