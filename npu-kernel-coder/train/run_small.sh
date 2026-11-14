#!/bin/sh
# Run train_small.py under a preemptible dibs GPU hold (same loop as run_train.sh): exit 75 hands
# the GPU over and dibs reruns it; exit 76 (memguard) waits for $RESUME_GB free, then requeues.
#   ./run_small.sh NAME [train_small.py flags]      output: runs/NAME/ (log: runs/NAME/out.log)
set -u
cd "$(dirname "$0")"
NAME=$1; shift
MODEL=${MODEL:-$(ls -d /home/cyrus/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/* | head -1)}
TRAIN=${TRAIN:-data/tasks/train.jsonl}
VAL=${VAL:-data/tasks/val.jsonl}
mkdir -p "runs/$NAME"
avail_gb() { awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
while :; do
  dibs run gpu --label "npu-kernel-coder small-model SFT $NAME (preemptible)" --eta "${ETA:-60m}" --preemptible --requeue -- sh -c "
    ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
    K=\$!
    trap 'kill \$K 2>/dev/null' EXIT
    PYTHONPATH=.fla .venv/bin/python train_small.py '$MODEL' '$TRAIN' '$VAL' --out 'runs/$NAME' $*
  " >> "runs/$NAME/out.log" 2>&1
  rc=$?
  [ "$rc" = 76 ] || exit "$rc"
  echo "run_small: memguard stop at $(date '+%F %T'); waiting for ${RESUME_GB:-30} GB free" >> "runs/$NAME/out.log"
  ok=0
  while [ "$ok" -lt 60 ]; do
    if [ "$(avail_gb)" -ge "${RESUME_GB:-30}" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 5
  done
done
