#!/usr/bin/env bash
# Continue the light pilot's fixed, overlap and joint arms for about one million more tokens each,
# from their final weights (fresh AdamW + warmup: the pilot did not save optimizer state), on the
# rows each arm had not yet drawn (--data-offset), same schedule, lr 3e-5, batch 1.
# Budgets are FLOPs that buy ~1M tokens at each arm's pilot rate of FLOPs per token.
# Then runs the evaluation driver (start student, continued arms, pilot arms).
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/continue-light.sh > healing-schedules/logs/continue-light.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
RUNS=$HS/runs/continue-light
BREAK_SECONDS=${BREAK_SECONDS:-600}
DONE=$HS/logs/continue-light.done
mkdir -p "$RUNS"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target-light.json'))['target_ce'])")
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")

done_run() { [ -f "$1/summary.json" ] && ! grep -q canary_failed "$1/summary.json"; }

# strategy  rows-already-drawn  budget(FLOPs for ~1M tokens)  eta
for spec in "fixed 483 6060000000000000 70m" "overlap 410 7110000000000000 80m" "joint 157 18650000000000000 150m"; do
  set -- $spec
  strategy=$1 offset=$2 budget=$3 eta=$4
  out=$RUNS/$strategy-seed17
  done_run "$out" && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  for attempt in 1 2 3; do
    log "$strategy: continuation attempt $attempt (offset $offset rows, budget $budget)"
    set +e
    dibs run gpu --label "healing-schedules light continue $strategy" --eta "$eta" -- \
      $PY $HS/run.py --teacher $TEACHER --damaged $HS/artifacts/continue-light-$strategy --data $DATA \
        --output "$out" --strategy $strategy --seed 17 --data-offset $offset --budget-flops $budget \
        --target-ce "$TARGET" --eval-points 8 --eval-sequences 32 --max-seconds 21600 \
        --batch-sequences 1 --lr 0.00003 "${CANARY[@]}"
    code=$?
    set -e
    [ $code -eq 0 ] && break
    [ $code -ne 3 ] && { log "$strategy: failed with exit $code"; exit $code; }
    log "$strategy: canary failed; rerunning"
    mv "$out" "$out.faulty-$attempt"
  done
  done_run "$out" || { log "$strategy: canary failed three times"; exit 3; }
  log "$strategy: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"
done
log "continuations complete"
touch "$DONE"
exec $HS/drivers/eval-pilot-light.sh
