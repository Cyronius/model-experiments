#!/usr/bin/env bash
# Canary reference, frozen baselines, frozen target, then the one-seed seven-arm pilot.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/pilot-mix-2048.sh > healing-schedules/logs/pilot-mix-2048.log 2>&1 < /dev/null & disown
# Each GPU job takes the GPU through dibs and is rerun in a fresh process (up to 3 attempts) when
# its canary fails (exit 3). Steps already done are skipped, so the script can be relaunched.
# Cancel: pkill -f pilot-mix-2048.sh (a running job finishes its current dibs command).
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
BASE=$HS/baselines/mix-2048
RUNS=$HS/runs/pilot-mix-2048
BUDGET=2000000000000000
DONE=$HS/logs/pilot-mix-2048.done
mkdir -p "$BASE" "$RUNS" "$HS/logs"
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }

retry_gpu() {  # retry_gpu <label> <eta> <output-path-to-move-aside-or-empty> -- cmd...
  local label=$1 eta=$2 out=$3; shift 4
  for attempt in 1 2 3; do
    log "$label: attempt $attempt"
    set +e
    dibs run gpu --label "$label" --eta "$eta" -- "$@"
    local code=$?
    set -e
    if [ $code -eq 0 ]; then return 0; fi
    if [ $code -ne 3 ]; then log "$label: failed with exit $code"; return $code; fi
    log "$label: canary failed; rerunning in a fresh process"
    if [ -n "$out" ] && [ -e "$out" ]; then mv "$out" "$out.faulty-$attempt"; fi
  done
  log "$label: canary failed three times"; return 3
}

if [ ! -f "$BASE/canary-cpu.json" ]; then
  log "CPU canary reference"
  dibs run cpu --label "healing-schedules canary reference" --eta 20m -- \
    $PY $HS/canary.py --teacher $TEACHER --data $DATA --output "$BASE/canary-cpu.json"
fi

for name in teacher native damaged; do
  case $name in
    teacher) model=$TEACHER ;;
    native) model=$HS/artifacts/native-0.8b ;;
    damaged) model=$HS/artifacts/damaged-0.8b ;;
  esac
  if [ ! -f "$BASE/$name.json" ]; then
    retry_gpu "healing-schedules baseline $name" 90m "" -- \
      $PY $HS/evaluate.py --model $model --teacher $TEACHER --tokenizer $TEACHER --data $DATA \
        --output "$BASE/$name.json" --canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json"
  fi
done

if [ ! -f "$BASE/target.json" ]; then
  $PY - "$BASE" <<'EOF'
import json, sys, datetime
from pathlib import Path
base = Path(sys.argv[1])
native = json.loads((base / 'native.json').read_text())['validation']
record = dict(target_ce=native['ce'], secondary_target_ce=native['ce'] + 0.10,
              rule='native Qwen3.5-0.8B validation CE on held-out biomed+argument; two consecutive evaluations at or below',
              native_validation=native, frozen_at=datetime.datetime.now().isoformat(timespec='seconds'))
(base / 'target.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record))
EOF
fi
TARGET=$($PY -c "import json;print(json.load(open('$BASE/target.json'))['target_ce'])")
log "frozen target CE $TARGET"

for strategy in joint layerwise fixed hierarchical overlap adaptive_error adaptive_plateau; do
  out=$RUNS/$strategy-seed17
  if [ -f "$out/summary.json" ] && ! grep -q canary_failed "$out/summary.json"; then continue; fi
  if [ -e "$out" ]; then mv "$out" "$out.incomplete-$(date +%s)"; fi
  retry_gpu "healing-schedules pilot $strategy" 60m "$out" -- \
    $PY $HS/run.py --teacher $TEACHER --damaged $HS/artifacts/damaged-0.8b --data $DATA \
      --output "$out" --strategy $strategy --seed 17 --budget-flops $BUDGET --target-ce "$TARGET" \
      --eval-points 5 --eval-sequences 16 --canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json"
done

log "pilot complete"
touch "$DONE"
