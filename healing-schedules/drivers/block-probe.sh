#!/usr/bin/env bash
# Block-level comparison with the teacher (block_probe.py): the damaged light student before healing
# against the 1x and 10x joint-cosine results. Did whole-network healing make the blocks resemble the
# teacher's again, or only the output?
# Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/block-probe.sh > healing-schedules/logs/block-probe.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
STUDENT=$HS/artifacts/damaged-light-h1792-n1473244736
OUT=$HS/runs/long-joint-cosine/block-probe.json
# The 1x run's weights stayed on the pre-reinstall disk.
ONE_X=${ONE_X:-/mnt/fedora/home/cyrus/code/model-experiments/$HS/runs/followup-random/joint-cosine-seed17/final}
BASE=$HS/baselines/mix-2048
KIND=(systemd-run --user --scope --quiet -p MemoryMax=24G -p CPUQuota=400% nice -n 10 ionice -c 3)
log() { echo "[$(date '+%F %T')] $*"; }

for attempt in 1 2 3; do
  set +e
  dibs run gpu --label "healing-schedules block probe (start, 1x, 10x)" --eta 15m -- "${KIND[@]}" \
    $PY $HS/block_probe.py --model "start=$STUDENT" --model "joint-cosine-1x=$ONE_X" \
      --model "joint-cosine-10x=$HS/runs/long-joint-cosine/joint-cosine-60e15-seed17/final" \
      --teacher models/manifold-audit/Qwen3.5-2B --pruning "$STUDENT/pruning.json" --data $HS/data/mix-2048 \
      --output "$OUT" --sequences 32 --canary "$BASE/canary-cpu.json" --canary-gpu "$BASE/canary-gpu.json"
  code=$?
  set -e
  [ $code -eq 0 ] && break
  [ $code -ne 3 ] && { log "block probe failed with exit $code"; exit $code; }
  log "canary failed; rerunning after a break"; sleep 300
done
[ -f "$OUT" ] || { log "canary failed three times"; exit 3; }
log "block probe complete"
touch $HS/logs/block-probe.done
