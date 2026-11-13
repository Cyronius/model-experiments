#!/usr/bin/env bash
# Score each v2 pilot arm's final model with the same evaluation as the baselines
# (corpus CE/KL, ARC-Easy, ARC-Challenge, MMLU, GSM8K at 1024 tokens), one GPU job per arm.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/eval-pilot-v2.sh > healing-schedules/logs/eval-pilot-v2.log 2>&1 < /dev/null & disown
# Finished arms are skipped, so the script can be relaunched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
RUNS=$HS/runs/pilot-v2
DONE=$HS/logs/eval-pilot-v2.done
BREAK_SECONDS=${BREAK_SECONDS:-600}
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }

for strategy in fixed adaptive_plateau hierarchical overlap joint layerwise adaptive_error; do
  run=$RUNS/$strategy-seed17
  out=$run/eval.json
  [ -f "$out" ] && continue
  [ -f "$run/final/config.json" ] || { log "$strategy: no final model"; exit 1; }
  for attempt in 1 2 3; do
    log "$strategy: evaluation attempt $attempt"
    set +e
    dibs run gpu --label "healing-schedules eval v2 $strategy" --eta 90m -- \
      $PY $HS/evaluate.py --model $run/final --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
        --output "$out" --gsm8k-tokens 1024 --canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json"
    code=$?
    set -e
    [ $code -eq 0 ] && break
    [ $code -ne 3 ] && { log "$strategy: failed with exit $code"; exit $code; }
    log "$strategy: canary failed; rerunning"
  done
  [ -f "$out" ] || { log "$strategy: canary failed three times"; exit 3; }
  log "$strategy: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"
done
log "all arms evaluated"
touch "$DONE"
