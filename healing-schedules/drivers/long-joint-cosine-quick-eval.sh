#!/usr/bin/env bash
# Quick (~20 min) evaluation of the long joint-cosine run, replacing the ~1 h full evaluation.
# Waits for long-joint-cosine.sh to finish training and enter its pre-evaluation GPU break, stops that
# driver (it is only sleeping then), and evaluates the final model on:
#   - the test split (validation CE is already the run's last curve point),
#   - the first 200 examples of ARC-Easy, ARC-Challenge and MMLU (fixed frozen order),
#   - the first 32 GSM8K problems at 1024 new tokens.
# Subsets are prefixes of the frozen sets, so earlier full evaluations can be rescored on exactly
# the same examples from their eval.json rows.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/long-joint-cosine-quick-eval.sh > healing-schedules/logs/long-joint-cosine-quick-eval.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
OUT=$HS/runs/long-joint-cosine/joint-cosine-60e15-seed17
LOG=$HS/logs/long-joint-cosine.log
DONE=$HS/logs/long-joint-cosine.done
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")
KIND=(systemd-run --user --scope --quiet -p MemoryMax=24G -p CPUQuota=400% nice -n 10 ionice -c 3)
log() { echo "[$(date '+%F %T')] $*"; }

until grep -q "evaluation start" "$LOG"; do
  grep -qE "failed with exit|failed three times" "$LOG" && { log "training driver failed; not evaluating"; exit 1; }
  sleep 30
done
driver=$(pgrep -f "drivers/long-joint-cosine[.]sh" || true)
if [ -n "$driver" ]; then
  log "stopping the full-evaluation driver ($driver) during its break"
  pkill -P "$driver" sleep || true
  kill $driver || true
  while kill -0 $driver 2>/dev/null; do sleep 2; done
fi
[ -f "$OUT/summary.json" ] || { log "no summary.json; training did not finish"; exit 1; }
log "GPU break, then quick evaluation"; sleep 300

for attempt in 1 2 3; do
  set +e
  dibs run gpu --label "healing-schedules long joint-cosine quick eval" --eta 30m -- "${KIND[@]}" \
    $PY $HS/evaluate.py --model "$OUT/final" --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
      --output "$OUT/eval-quick.json" --splits test --mc-examples 200 --gsm8k-examples 32 --gsm8k-tokens 1024 \
      "${CANARY[@]}"
  code=$?
  set -e
  [ $code -eq 0 ] && break
  [ $code -ne 3 ] && { log "evaluation failed with exit $code"; exit $code; }
  log "evaluation canary failed; rerunning"
done
[ -f "$OUT/eval-quick.json" ] || { log "evaluation canary failed three times"; exit 3; }
log "quick evaluation complete"
touch "$DONE"
