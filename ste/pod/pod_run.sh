#!/bin/bash
# Full data session, flag-gated so a rerun resumes. Hard-stops near 85 percent of BUDGET.
# Usage: RATE=0.79 BUDGET=5 N=12000 bash pod_run.sh
source "$(dirname "$0")/common.sh"
N=${N:-12000}
guard env;     phase_env || fail env
guard model;   fetch_model vol "$VOL_REPO" "*Q4_K_M*"; VOL=$MODEL_PATH
if ! done_flag sources; then
  guard sources
  python3 -I "$W/repo/ste/data/fetch_sources.py" --out "$OUT/sources.jsonl" --n "$N" --md-dir "$W/repo" | tee -a "$LOG" || fail sources
  mark sources
fi
guard server;  start_server "$VOL" 8080 || fail server
if ! done_flag seed; then
  guard seed; log "=== seed (volume teacher, up to 4 tries each) ==="
  gen rewrite --url http://127.0.0.1:8080 --input "$OUT/sources.jsonl" --out "$OUT/seed.jsonl" --workers "$SLOTS" --retries 4 | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark seed || log "seed cut short (budget or error); rerun resumes it"
fi
if ! done_flag hard; then
  guard hard; log "=== hard (seed failures, 8 tries, hotter) ==="
  python3 -I - <<PY
import json
rows=[json.loads(l) for l in open("$OUT/seed.jsonl.failed")] if __import__("os").path.exists("$OUT/seed.jsonl.failed") else []
with open("$OUT/hard-in.jsonl","w") as f:
    for r in rows: f.write(json.dumps({"id": r["id"], "text": r["src"], "mode": r["mode"]})+"\n")
print(len(rows), "hard inputs")
PY
  gen rewrite --url http://127.0.0.1:8080 --input "$OUT/hard-in.jsonl" --out "$OUT/hard.jsonl" --workers "$SLOTS" --retries 8 --temperature 0.9 | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark hard || log "hard cut short; rerun resumes it"
fi
if ! done_flag messy; then
  guard messy; log "=== messy back-translation ==="
  cat "$OUT/seed.jsonl" "$OUT/hard.jsonl" > "$OUT/clean.jsonl" 2>/dev/null
  gen messy --url http://127.0.0.1:8080 --input "$OUT/clean.jsonl" --out "$OUT/messy.jsonl" --workers "$SLOTS" | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark messy || log "messy cut short; rerun resumes it"
fi
if ! done_flag judge; then
  guard judge; fetch_model judge "$JUDGE_REPO" "*mxfp4*"; JUDGE=$MODEL_PATH
  start_server "$JUDGE" 8081 || fail judge
  python3 -I "$W/repo/ste/data/judge.py" --url http://127.0.0.1:8081 --input "$OUT/clean.jsonl" --out "$OUT/judged.jsonl" --limit 1500 --workers "$SLOTS" | tee -a "$LOG"
  mark judge
fi
log "ALL PHASES DONE"; pack; stop_pod
