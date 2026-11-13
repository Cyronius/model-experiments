#!/usr/bin/env bash
# Follow-up 3 on the light (25%) student, one seed, same budget and optimizer as the light pilot
# (6e15 training FLOPs ~ 1M tokens for fixed, lr 3e-5, batch 1), compared with the pilot's arms:
#   global_fixed        whole network for 1/7 of the budget, then fixed blocks (1/7 each)
#   alternating         fixed sweep; the last 20% of each of 24 FLOP slices trains the whole network
#   fixed-teacher       fixed, each block fed the teacher's stream (no student prefix)
#   fixed-ramp          fixed, teacher-fed with probability falling 1 -> 0 inside each block
# Then the baseline-format evaluation of each final model.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/followup-local-global.sh > healing-schedules/logs/followup-local-global.log 2>&1 < /dev/null & disown
# Finished runs and evaluations are skipped, so the script can be relaunched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
STUDENT=$HS/artifacts/damaged-light-h1792-n1473244736
RUNS=$HS/runs/followup-local-global
BREAK_SECONDS=${BREAK_SECONDS:-600}
DONE=$HS/logs/followup-local-global.done
mkdir -p "$RUNS"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target-light.json'))['target_ce'])")
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")
ARMS=("global_fixed global_fixed student" "alternating alternating student"
      "fixed-teacher fixed teacher" "fixed-ramp fixed ramp")

done_run() { [ -f "$1/summary.json" ] && ! grep -q canary_failed "$1/summary.json"; }

gpu() {  # gpu <label> <eta> <move-aside-on-canary-failure> -- cmd...
  local label=$1 eta=$2 out=$3; shift 4
  for attempt in 1 2 3; do
    log "$label: attempt $attempt"
    set +e
    dibs run gpu --label "$label" --eta "$eta" -- "$@"
    local code=$?
    set -e
    if [ $code -eq 0 ]; then log "$label: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"; return 0; fi
    if [ $code -ne 3 ]; then log "$label: failed with exit $code"; exit $code; fi
    log "$label: canary failed; rerunning"
    [ -n "$out" ] && [ -e "$out" ] && mv "$out" "$out.faulty-$attempt"
  done
  log "$label: canary failed three times"; exit 3
}

for arm in "${ARMS[@]}"; do
  set -- $arm
  name=$1 strategy=$2 inputs=$3
  out=$RUNS/$name-seed17
  done_run "$out" && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  gpu "healing-schedules followup $name" 80m "$out" -- \
    $PY $HS/run.py --teacher $TEACHER --damaged $STUDENT --data $DATA --output "$out" \
      --strategy $strategy --window-inputs $inputs --seed 17 --budget-flops 6000000000000000 \
      --target-ce "$TARGET" --eval-points 8 --eval-sequences 32 --max-seconds 21600 \
      --batch-sequences 1 --lr 0.00003 "${CANARY[@]}"
done

for arm in "${ARMS[@]}"; do
  set -- $arm
  run=$RUNS/$1-seed17
  [ -f "$run/eval.json" ] && continue
  gpu "healing-schedules eval followup $1" 120m "" -- \
    $PY $HS/evaluate.py --model "$run/final" --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
      --output "$run/eval.json" --gsm8k-tokens 1024 "${CANARY[@]}"
done

log "follow-up 3 complete"
touch "$DONE"
