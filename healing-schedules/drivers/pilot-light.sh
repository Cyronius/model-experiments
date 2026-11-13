#!/usr/bin/env bash
# Light-damage study (25% non-embedding cut, hidden 1792 / FFN 4800), one seed:
#   1. optimizer check: joint at three settings; the lowest final controller CE picks the
#      batch/learning rate (controller split, so validation stays clean for the comparison);
#   2. pilot: all seven schedules with the chosen settings.
# The 100% rung is the 2B teacher's validation CE (no stock model exists at this size).
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/pilot-light.sh > healing-schedules/logs/pilot-light.log 2>&1 < /dev/null & disown
# Steps already done are skipped, so the script can be relaunched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
STUDENT=$HS/artifacts/damaged-light-h1792-n1473244736
CHECK=$HS/runs/light-optcheck
RUNS=$HS/runs/pilot-light
BUDGET=${BUDGET:-6000000000000000}
BREAK_SECONDS=${BREAK_SECONDS:-600}
DONE=$HS/logs/pilot-light.done
mkdir -p "$CHECK" "$RUNS"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
CANARY=(--canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json")
COMMON=(--teacher $TEACHER --damaged $STUDENT --data $DATA --seed 17 --budget-flops $BUDGET
        --eval-points 8 --eval-sequences 32 --max-seconds 10800)

retry_gpu() {  # retry_gpu <label> <eta> <output-dir> -- cmd...
  local label=$1 eta=$2 out=$3; shift 4
  for attempt in 1 2 3; do
    log "$label: attempt $attempt"
    set +e
    dibs run gpu --label "$label" --eta "$eta" -- "$@"
    local code=$?
    set -e
    if [ $code -eq 0 ]; then log "$label: done; ${BREAK_SECONDS}s GPU break"; sleep "$BREAK_SECONDS"; return 0; fi
    if [ $code -ne 3 ]; then log "$label: failed with exit $code"; return $code; fi
    log "$label: canary failed; rerunning in a fresh process"
    [ -e "$out" ] && mv "$out" "$out.faulty-$attempt"
  done
  log "$label: canary failed three times"; return 3
}

done_run() { [ -f "$1/summary.json" ] && ! grep -q canary_failed "$1/summary.json"; }

# Wait (up to 2 h) for the CPU build (drivers/light-probes.sh) to finish.
for _ in $(seq 240); do [ -f "$STUDENT/pruning.json" ] && break; sleep 30; done
[ -f "$STUDENT/pruning.json" ] || { log "missing $STUDENT"; exit 1; }
if [ ! -f "$BASE/target-light.json" ]; then
  $PY - "$BASE" <<'EOF'
import json, sys, datetime
from pathlib import Path
base = Path(sys.argv[1])
teacher = json.loads((base / 'teacher.json').read_text())['validation']
record = dict(target_ce=teacher['ce'], rule='2B teacher validation CE on held-out biomed+argument (light-damage study; no stock model at 1.47B); two consecutive evaluations at or below',
              teacher_validation={k: teacher[k] for k in ('ce', 'perplexity', 'tokens')},
              frozen_at=datetime.datetime.now().isoformat(timespec='seconds'))
(base / 'target-light.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record))
EOF
fi
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target-light.json'))['target_ce'])")
log "light-damage target CE (teacher) $TARGET; budget $BUDGET per run"

# 1. Optimizer check on joint.
for setting in "b1-lr1e-4 1 0.0001" "b1-lr3e-5 1 0.00003" "b4-lr1e-4 4 0.0001"; do
  set -- $setting
  out=$CHECK/joint-$1-seed17
  done_run "$out" && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  retry_gpu "healing-schedules light optcheck $1" 60m "$out" -- \
    $PY $HS/run.py "${COMMON[@]}" --output "$out" --strategy joint --target-ce "$TARGET" \
      --batch-sequences $2 --lr $3 "${CANARY[@]}"
done
CHOICE=$($PY - "$CHECK" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for run in sorted(Path(sys.argv[1]).glob('joint-*-seed17')):
    if not (run / 'summary.json').exists():
        continue
    curve = json.loads((run / 'curve.json').read_text())
    args = json.loads((run / 'provenance.json').read_text())['arguments']
    rows.append(dict(run=run.name, controller_ce=curve[-1]['controller']['ce'], validation_ce=curve[-1]['validation']['ce'],
                     batch_sequences=args['batch_sequences'], lr=args['lr']))
best = min(rows, key=lambda r: r['controller_ce'])
(Path(sys.argv[1]) / 'choice.json').write_text(json.dumps(dict(chosen=best, candidates=rows,
    rule='lowest final controller CE (in-mix held-out); validation is reported, not used'), indent=1) + '\n')
print(best['batch_sequences'], best['lr'], best['run'])
EOF
)
set -- $CHOICE
BATCH=$1 LR=$2 CHOSEN=$3
log "optimizer choice: batch $BATCH, lr $LR ($CHOSEN)"

# 2. Seven-arm pilot.
for strategy in fixed adaptive_plateau hierarchical overlap joint layerwise adaptive_error; do
  out=$RUNS/$strategy-seed17
  done_run "$out" && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  # The chosen optimizer-check run is the joint arm (identical arguments apart from --output).
  if [ $strategy = joint ]; then cp -a "$CHECK/$CHOSEN" "$out"; log "joint: reused $CHOSEN"; continue; fi
  retry_gpu "healing-schedules light pilot $strategy" 70m "$out" -- \
    $PY $HS/run.py "${COMMON[@]}" --output "$out" --strategy $strategy --target-ce "$TARGET" \
      --batch-sequences $BATCH --lr $LR "${CANARY[@]}"
done

log "light pilot complete"
touch "$DONE"
