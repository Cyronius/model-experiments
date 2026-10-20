#!/bin/bash
# Full data session, flag-gated so a rerun resumes. Hard-stops near 85 percent of BUDGET.
# Usage: RATE=1.09 BUDGET=3 N=12000 bash pod_run.sh
# Order: seed and hard rewrites (vLLM), judge every row (llama.cpp), then messy back-translation
# of only the rows the judge rates 5, so no teacher time goes to rows we will drop.
source "$(dirname "$0")/common.sh"
N=${N:-12000}
guard env;     phase_env || fail env
guard vllm;    phase_vllm || fail vllm
if ! done_flag sources; then
  guard sources
  python3 -I "$W/repo/asd-ste100/data/fetch_sources.py" --out "$OUT/sources.jsonl" --n "$N" --md-dir "$W/repo" | tee -a "$LOG" || fail sources
  mark sources
fi
if ! done_flag seed || ! done_flag hard; then
  guard server; start_vllm "$TEACHER" 8080 || fail server
fi
if ! done_flag seed; then
  guard seed; log "=== seed (up to 4 tries each) ==="
  gen rewrite --url http://127.0.0.1:8080 --input "$OUT/sources.jsonl" --out "$OUT/seed.jsonl" --workers "$VSLOTS" --retries 4 | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark seed || log "seed cut short (budget or error); rerun resumes it"
fi
if ! done_flag hard; then
  guard hard; log "=== hard (seed failures, 8 tries, hotter) ==="
  python3 -I - <<PY
import json, os
p = "$OUT/seed.jsonl.failed"
rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
with open("$OUT/hard-in.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"id": r["id"], "text": r["src"], "mode": r["mode"]}) + "\n")
print(len(rows), "hard inputs")
PY
  gen rewrite --url http://127.0.0.1:8080 --input "$OUT/hard-in.jsonl" --out "$OUT/hard.jsonl" --workers "$VSLOTS" --retries 8 --temperature 0.9 | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark hard || log "hard cut short; rerun resumes it"
fi
stop_vllm
if ! done_flag judge; then
  guard judge; log "=== judge every clean row ==="
  cat "$OUT/seed.jsonl" "$OUT/hard.jsonl" > "$OUT/clean.jsonl" 2>/dev/null
  fetch_model judge "$JUDGE_REPO" "gpt-oss-20b-MXFP4.gguf"; JUDGE=$MODEL_PATH
  start_server "$JUDGE" 18081 || fail judge
  timeout "$(remaining)" python3 -I "$W/repo/asd-ste100/data/judge.py" --url http://127.0.0.1:18081 --input "$OUT/clean.jsonl" --out "$OUT/judged.jsonl" --workers "$SLOTS" | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark judge || log "judge cut short; rerun resumes it"
  pkill -x llama-server; sleep 5
fi
python3 -I - <<PY | tee -a "$LOG"
import json
rows = [json.loads(l) for l in open("$OUT/judged.jsonl")]
keep = [r for r in rows if r.get("score") == 5]
with open("$OUT/clean5.jsonl", "w") as f:
    for r in keep: f.write(json.dumps({k: r[k] for k in ("id", "src", "ste", "mode")}, ensure_ascii=False) + "\n")
print(f"judge kept {len(keep)} of {len(rows)} rows (score 5) -> clean5.jsonl")
PY
pack
if ! done_flag messy; then
  guard messy; log "=== messy back-translation of score-5 rows ==="
  start_vllm "$TEACHER" 8080 || fail messy
  gen messy --url http://127.0.0.1:8080 --input "$OUT/clean5.jsonl" --out "$OUT/messy.jsonl" --workers "$VSLOTS" | tail -1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && mark messy || log "messy cut short; rerun resumes it"
  stop_vllm
fi
log "ALL PHASES DONE"; pack; stop_pod
