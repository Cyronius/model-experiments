#!/usr/bin/env bash
# Light-damage follow-up: build three lighter activation-selected students on the CPU and probe
# their start CE, to choose the cut. No GPU.
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/light-probes.sh > healing-schedules/logs/light-probes.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
TEACHER=models/manifold-audit/Qwen3.5-2B
DATA=$HS/data/mix-2048
DONE=$HS/logs/light-probes.done
rm -f "$DONE"
log() { echo "[$(date '+%F %T')] $*"; }
# Cuts of at most 25% of non-embedding parameters (teacher 1,373,265,728), from accounting.count:
#   hidden 1792 / FFN 4800: 25.1% cut, 1.473B total
#   hidden 1792 / FFN 5376: 19.7% cut, 1.548B total
#   hidden 1920 / FFN 5248: 15.3% cut, 1.640B total
for spec in "1792 1473244736" "1792 1547562560" "1920 1640376000"; do
  set -- $spec
  out=$HS/artifacts/damaged-light-h$1-n$2
  [ -f "$out/pruning.json" ] && continue
  [ -e "$out" ] && mv "$out" "$out.incomplete-$(date +%s)"
  log "build hidden $1 (target $2 parameters) and probe its start CE"
  dibs run cpu --label "healing-schedules light-damage probe h$1 n$2" --eta 45m -- \
    $PY $HS/pruning.py --source $TEACHER --output $out --data $DATA --hidden $1 --target $2
done
log "light probes complete"
touch "$DONE"
