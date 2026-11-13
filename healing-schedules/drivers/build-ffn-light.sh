#!/usr/bin/env bash
# Conversion-order follow-up (planned): an FFN-only 24.7% non-embedding cut of the teacher
# (hidden 2048 kept, FFN 6144 -> 3840), so any layer can stay the teacher's own until converted.
# CPU only. Launch detached from the repository root:
#   setsid nohup healing-schedules/drivers/build-ffn-light.sh > healing-schedules/logs/build-ffn-light.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=target/manifold-rocm/env/bin/python
HS=healing-schedules
OUT=$HS/artifacts/damaged-ffn-h2048-n1542086464
[ -f "$OUT/pruning.json" ] && exit 0
[ -e "$OUT" ] && mv "$OUT" "$OUT.incomplete-$(date +%s)"
dibs run cpu --label "healing-schedules FFN-only light student build" --eta 60m -- \
  $PY $HS/pruning.py --source models/manifold-audit/Qwen3.5-2B --output $OUT --data $HS/data/mix-2048 \
    --hidden 2048 --target 1542086464
touch $HS/logs/build-ffn-light.done
