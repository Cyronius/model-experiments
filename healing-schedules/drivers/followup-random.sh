#!/usr/bin/env bash
# Follow-up 4 on the light (25%) student, one seed, same budget and optimizer as the light pilot
# (6e15 training FLOPs, lr 3e-5, batch 1), but 24 evaluation points (every 0.25e15, a superset of
# the pilot's 8) so FLOPs to the 25% rung is resolved 3x finer:
#   joint-cosine        joint with the learning rate cosine-decayed to 0 over the budget
#   random              a fresh random window every step (width uniform 1..24, start uniform)
#   random_dwell        random windows kept until their own loss gains 5%, stalls or hits 60 steps
#   coarse_to_fine      hierarchical reversed: whole network, then widths 16, 8, 4, 2, 1
#                       (ties joint to 25% by construction; measured for later rungs and final quality)
# Then the baseline-format evaluation of each final model.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/followup-random.sh > healing-schedules/logs/followup-random.log 2>&1 < /dev/null & disown
# Finished runs and evaluations are skipped, so the script can be relaunched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
STUDENT=$HS/artifacts/damaged-light-h1792-n1473244736
RUNS=$HS/runs/followup-random
BREAK_SECONDS=${BREAK_SECONDS:-600}
DONE=$HS/logs/followup-random.done
mkdir -p "$RUNS"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target-light.json'))['target_ce'])")
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")
ARMS=("joint-cosine joint cosine" "random random none" "random_dwell random_dwell none"
      "coarse_to_fine coarse_to_fine none")

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
  name=$1 strategy=$2 decay=$3
  out=$RUNS/$name-seed17
  done_run "$out" && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  gpu "healing-schedules followup $name" 110m "$out" -- \
    $PY $HS/run.py --teacher $TEACHER --damaged $STUDENT --data $DATA --output "$out" \
      --strategy $strategy --lr-decay $decay --seed 17 --budget-flops 6000000000000000 \
      --target-ce "$TARGET" --eval-points 24 --eval-sequences 32 --max-seconds 21600 \
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

log "follow-up 4 complete"
touch "$DONE"
