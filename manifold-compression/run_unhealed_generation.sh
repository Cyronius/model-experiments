#!/usr/bin/env bash
# Frozen G1 run order: P0, P1, then stock; parts track-a, gsm8k, humaneval, science-new.
# A failed or timed-out part is logged and the rest still run until the shared clock stops them;
# any incomplete part leaves the screen without a progression decision. Logs hold only
# progress counts and exit codes, never answers or grades.
set -u
cd "$(dirname "$0")" || exit 1
PY=../target/manifold-audit/env/bin/python
LOG=results/unhealed-generation-20261027/logs
mkdir -p "$LOG" || exit 1
for arm in P0 P1 stock; do
  for part in track-a gsm8k humaneval science-new; do
    $PY unhealed_generation.py generate --arm "$arm" --part "$part" > "$LOG/$arm-$part.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) $arm $part exit $rc" | tee -a "$LOG/run.log"
  done
done
