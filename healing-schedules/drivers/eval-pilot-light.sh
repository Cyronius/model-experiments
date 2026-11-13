#!/usr/bin/env bash
# Score the light-damage student (start) and each light pilot arm's final model with the same
# evaluation as the baselines (corpus CE/KL, ARC-Easy, ARC-Challenge, MMLU, GSM8K at 1024 tokens),
# one GPU job per model: start student, continued arms (drivers/continue-light.sh), then pilot arms
# by final validation loss.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/eval-pilot-light.sh > healing-schedules/logs/eval-pilot-light.log 2>&1 < /dev/null & disown
# Finished models are skipped, so the script can be relaunched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
RUNS=$HS/runs/pilot-light
DONE=$HS/logs/eval-pilot-light.done
BREAK_SECONDS=${BREAK_SECONDS:-600}
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }

evaluate() {  # evaluate <name> <model-dir> <output>
  local name=$1 model=$2 out=$3
  [ -f "$out" ] && return 0
  [ -f "$model/config.json" ] || { log "$name: no model at $model"; exit 1; }
  for attempt in 1 2 3; do
    log "$name: evaluation attempt $attempt"
    set +e
    dibs run gpu --label "healing-schedules eval light $name" --eta 180m -- \
      $PY $HS/evaluate.py --model "$model" --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
        --output "$out" --gsm8k-tokens 1024 --canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json"
    local code=$?
    set -e
    [ $code -eq 0 ] && break
    [ $code -ne 3 ] && { log "$name: failed with exit $code"; exit $code; }
    log "$name: canary failed; rerunning"
  done
  [ -f "$out" ] || { log "$name: canary failed three times"; exit 3; }
  log "$name: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"
}

evaluate damaged-light $HS/artifacts/damaged-light-h1792-n1473244736 "$BASE/damaged-light.json"
for strategy in fixed overlap joint; do
  evaluate "continued-$strategy" "$HS/runs/continue-light/$strategy-seed17/final" "$HS/runs/continue-light/$strategy-seed17/eval.json"
done
for strategy in fixed overlap hierarchical layerwise adaptive_plateau adaptive_error joint; do
  evaluate "$strategy" "$RUNS/$strategy-seed17/final" "$RUNS/$strategy-seed17/eval.json"
done

log "light evaluations complete"
touch "$DONE"
