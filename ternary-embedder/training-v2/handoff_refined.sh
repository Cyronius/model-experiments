#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ternary-pilot
while [[ ! -s ternary-embedder/training-v2/work/balanced-prompted.json ]]; do sleep 1; done
kill -TERM -- -"$(cat pilot.pid)" 2>/dev/null || true
sleep 2
mv pilot.log ternary-embedder/training-v2/runs/initial-coverage-encoding.log
echo running > ternary-embedder/training-v2/pilot.status
(nohup timeout 70m bash ternary-embedder/training-v2/pod_entry.sh run_refined_round2.sh > pilot.log 2>&1 < /dev/null & echo $! > pilot.pid)
touch ternary-embedder/training-v2/work/refined-stage.started
