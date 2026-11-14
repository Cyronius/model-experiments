#!/bin/bash
# Continue the saved 200M student to 500M; all training outputs are on the pod volume.
set -euo pipefail
W=/workspace
MC=$W/model-experiments/manifold-compression
OUT=$W/distill-500m
mkdir -p "$W/logs" "$OUT"
rm -f "$W/logs/distill-500m.failed" "$W/logs/distill-500m.done" "$W/logs/distill-500m.exit"
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8
cd "$MC"
printf 'starting 500M continuation at %s\n' "$(date -u +%FT%TZ)"
if [ -f "$OUT/resume_latest.pt" ]; then
  EXTRA=(--resume-state "$OUT/resume_latest.pt")
else
  EXTRA=(--start-tokens 200015872 --history "$W/original-history.json")
fi
set +e
timeout --signal=TERM --kill-after=600s 32h python -u distill_whole.py \
  --student "$W/student-200M" --out "$OUT" --tokens 500e6 \
  --rows-cache "$W/rows-500m.pt" --checkpoints 250e6,300e6,350e6,400e6,450e6,500e6 \
  --save-every 10e6 "${EXTRA[@]}" > "$W/logs/distill-500m.log" 2>&1
status=$?
printf '%s\n' "$status" > "$W/logs/distill-500m.exit"
if [ "$status" = 0 ]; then
  touch "$W/logs/distill-500m.done"
else
  touch "$W/logs/distill-500m.failed"
fi
exit "$status"
