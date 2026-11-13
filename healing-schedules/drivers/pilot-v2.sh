#!/usr/bin/env bash
# v2 pilot: activation-selected native-size student, trainable embedding/head/final norm,
# per-group warmup, block-based adaptive controllers; then baselines with GSM8K at 1024 tokens.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/pilot-v2.sh > healing-schedules/logs/pilot-v2.log 2>&1 < /dev/null & disown
# Each GPU job is its own dibs run (pilot arms ~40 min) followed by a 10-minute break.
# Reuses the v1 canary references and frozen target (same teacher, data and native model).
# Steps already done are skipped, so the script can be relaunched. Cancel: pkill -f pilot-v2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
STUDENT=$HS/artifacts/damaged-v2-0.8b
RUNS=$HS/runs/pilot-v2
BUDGET=6000000000000000
DONE=$HS/logs/pilot-v2.done
mkdir -p "$RUNS" "$HS/logs"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
BREAK_SECONDS=${BREAK_SECONDS:-600}
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")

retry_gpu() {  # retry_gpu <label> <eta> <output-path-to-move-aside-or-empty> -- cmd...
  local label=$1 eta=$2 out=$3; shift 4
  for attempt in 1 2 3; do
    log "$label: attempt $attempt"
    set +e
    dibs run gpu --label "$label" --eta "$eta" -- "$@"
    local code=$?
    set -e
    if [ $code -eq 0 ]; then
      # Take turns: leave the GPU free between jobs so other sessions can get it.
      log "$label: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"; return 0
    fi
    if [ $code -ne 3 ]; then log "$label: failed with exit $code"; return $code; fi
    log "$label: canary failed; rerunning in a fresh process"
    if [ -n "$out" ] && [ -e "$out" ]; then mv "$out" "$out.faulty-$attempt"; fi
  done
  log "$label: canary failed three times"; return 3
}

for f in canary-cpu.json canary-gpu.json target.json; do
  [ -f "$BASE/$f" ] || { log "missing $BASE/$f (run drivers/pilot-mix-2048.sh first)"; exit 1; }
done

if [ ! -f "$STUDENT/pruning.json" ]; then
  [ -e "$STUDENT" ] && mv "$STUDENT" "$STUDENT.incomplete-$(date +%s)"
  log "build activation-selected student (CPU) and probe its start quality"
  dibs run cpu --label "healing-schedules v2 pruning" --eta 150m -- \
    $PY $HS/pruning.py --source $TEACHER --output $STUDENT --data $DATA
fi

TARGET=$($PY -c "import json;print(json.load(open('$BASE/target.json'))['target_ce'])")
log "frozen target CE $TARGET"

for strategy in joint layerwise fixed hierarchical overlap adaptive_error adaptive_plateau; do
  out=$RUNS/$strategy-seed17
  if [ -f "$out/summary.json" ] && ! grep -q canary_failed "$out/summary.json"; then continue; fi
  if [ -e "$out" ]; then mv "$out" "$out.incomplete-$(date +%s)"; fi
  retry_gpu "healing-schedules v2 pilot $strategy" 50m "$out" -- \
    $PY $HS/run.py --teacher $TEACHER --damaged $STUDENT --data $DATA \
      --output "$out" --strategy $strategy --seed 17 --budget-flops $BUDGET --target-ce "$TARGET" \
      --eval-points 6 --eval-sequences 16 --max-seconds 7200 "${CANARY[@]}"
done

# Baselines with GSM8K at 1024 new tokens (v1 used 384; 74 of 128 replies were cut off).
for name in damaged-v2 teacher-gsm1024 native-gsm1024; do
  case $name in
    damaged-v2) model=$STUDENT ;;
    teacher-gsm1024) model=$TEACHER ;;
    native-gsm1024) model=$HS/artifacts/native-0.8b ;;
  esac
  if [ ! -f "$BASE/$name.json" ]; then
    retry_gpu "healing-schedules baseline $name" 180m "" -- \
      $PY $HS/evaluate.py --model $model --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
        --output "$BASE/$name.json" --gsm8k-tokens 1024 "${CANARY[@]}"
  fi
done

log "v2 pilot complete"
touch "$DONE"
