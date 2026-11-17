#!/usr/bin/env bash
# How far does joint-cosine keep improving? The light (25%) student's best arm, run for 10x the
# study budget: 60e15 training FLOPs (about 3.2M tokens), learning rate 3e-5 cosine-decayed to 0
# over the whole budget, batch 1, seed 17, 40 evaluations of 32 sequences (every 1.5e15).
#
# Kind to the shared machine:
#   - each segment trains for about 2 h (--pause-after), saves exact resume state, exits 75 and
#     releases the GPU for BREAK_SECONDS before the next segment resumes it (bit-identical to an
#     uninterrupted run; test_pause_and_resume_matches_uninterrupted_run);
#   - nice 10 / idle I/O class, 2 torch CPU threads, 400% CPU quota and a host-memory cap via a
#     user systemd scope.
# Then the baseline-format evaluation of the final model.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/long-joint-cosine.sh > healing-schedules/logs/long-joint-cosine.log 2>&1 < /dev/null & disown
# Relaunching continues from the resume state.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
STUDENT=$HS/artifacts/damaged-light-h1792-n1473244736
OUT=$HS/runs/long-joint-cosine/joint-cosine-60e15-seed17
BREAK_SECONDS=${BREAK_SECONDS:-600}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-7200}
DONE=$HS/logs/long-joint-cosine.done
mkdir -p "$(dirname "$OUT")"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target-light.json'))['target_ce'])")
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")
KIND=(systemd-run --user --scope --quiet -p MemoryMax=24G -p CPUQuota=400% nice -n 10 ionice -c 3)

gpu() {  # gpu <label> <eta> -- cmd...; returns the command's exit code (call between set +e / set -e)
  local label=$1 eta=$2; shift 3
  dibs run gpu --label "$label" --eta "$eta" -- "${KIND[@]}" "$@"
}

canary_failures=0
while [ ! -f "$OUT/summary.json" ]; do
  resume=()
  if [ -f "$OUT/resume/state.pt" ]; then
    resume=(--resume)
  elif [ -e "$OUT" ]; then
    log "no resume state; moving the unfinished start aside"
    mv "$OUT" "$OUT.incomplete-$(date +%s)"
  fi
  log "segment start ${resume[*]:-(fresh)}"
  set +e
  gpu "healing-schedules long joint-cosine (segment of ~2h, yields between)" 140m -- \
    $PY $HS/run.py --teacher $TEACHER --damaged $STUDENT --data $DATA --output "$OUT" \
      --strategy joint --lr-decay cosine --seed 17 --budget-flops 60000000000000000 \
      --target-ce "$TARGET" --eval-points 40 --eval-sequences 32 --max-seconds 400000 \
      --batch-sequences 1 --lr 0.00003 --pause-after "$SEGMENT_SECONDS" "${resume[@]}" "${CANARY[@]}"
  code=$?
  set -e
  case $code in
    0) log "training finished" ;;
    75) log "segment paused; ${BREAK_SECONDS}s GPU break"; canary_failures=0; sleep "$BREAK_SECONDS" ;;
    3) canary_failures=$((canary_failures + 1)); log "canary failed ($canary_failures); retrying from the last resume state"
       [ $canary_failures -ge 3 ] && { log "canary failed three times in a row"; exit 3; }
       sleep "$BREAK_SECONDS" ;;
    *) log "failed with exit $code"; exit $code ;;
  esac
done

if [ ! -f "$OUT/eval.json" ]; then
  log "evaluation start; ${BREAK_SECONDS}s GPU break first"; sleep "$BREAK_SECONDS"
  for attempt in 1 2 3; do
    set +e
    gpu "healing-schedules long joint-cosine eval" 150m -- \
      $PY $HS/evaluate.py --model "$OUT/final" --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
        --output "$OUT/eval.json" --gsm8k-tokens 1024 "${CANARY[@]}"
    code=$?
    set -e
    [ $code -eq 0 ] && break
    [ $code -ne 3 ] && { log "evaluation failed with exit $code"; exit $code; }
    log "evaluation canary failed; rerunning"
  done
  [ -f "$OUT/eval.json" ] || { log "evaluation canary failed three times"; exit 3; }
fi
log "long joint-cosine complete"
touch "$DONE"
