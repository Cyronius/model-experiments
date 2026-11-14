#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ternary-pilot
while [[ "$(cat ternary-embedder/training-v2/pilot.status)" != done ]]; do
  if [[ "$(cat ternary-embedder/training-v2/pilot.status)" == failed ]]; then exit 1; fi
  sleep 1
done
echo running > ternary-embedder/training-v2/pilot.status
(nohup timeout 30m bash ternary-embedder/training-v2/pod_entry.sh run_round2_teacher_control.sh >> pilot.log 2>&1 < /dev/null & echo $! > pilot.pid)
touch ternary-embedder/training-v2/work/teacher-control.started
