#!/bin/bash
# 200-request probe: teacher speed and success rate, then a judge score on the accepted rows.
# Intended spend: about $1 to $2. Usage: RATE=1.09 BUDGET=2 bash pod_smoke.sh
source "$(dirname "$0")/common.sh"
guard env;  phase_env || fail env
guard vllm; phase_vllm || fail vllm
guard sources
python3 -I "$W/repo/asd-ste100/data/fetch_sources.py" --out "$OUT/sources.jsonl" --n 3000 --md-dir "$W/repo" | tee -a "$LOG"
guard server; start_vllm "$TEACHER" 8080 || fail server
guard gen
T0=$(date +%s)
gen rewrite --url http://127.0.0.1:8080 --input "$OUT/sources.jsonl" --out "$OUT/smoke-seed.jsonl" --workers "$VSLOTS" --limit 200 | tail -1 | tee "$OUT/smoke.json"
T1=$(date +%s)
python3 - <<PY | tee -a "$LOG"
import json
r = json.load(open("$OUT/smoke.json")); secs = max($T1 - $T0, 1)
n = r["ok"] + r["failed"]; per_req = secs / max(r["ok"], 1)
tps = r["gen_tokens"] / secs
print(f"smoke: {n} requests in {secs}s = {per_req:.2f}s per accepted row, {tps:.0f} generated tok/s, success {r['ok']}/{n}")
for total in (12000, 30000):
    h = total * per_req / 3600
    print(f"  {total} accepted rows: about {h:.1f} h generation, about \${h * $RATE:.2f} at \${$RATE}/h (before messy, hard, judge phases)")
PY
stop_vllm
guard judge; fetch_model judge "$JUDGE_REPO" "gpt-oss-20b-MXFP4.gguf"; JUDGE=$MODEL_PATH
start_server "$JUDGE" 18081 || fail judge
python3 -I "$W/repo/asd-ste100/data/judge.py" --url http://127.0.0.1:18081 --input "$OUT/smoke-seed.jsonl" --out "$OUT/smoke-judged.jsonl" --limit 200 --workers "$SLOTS" | tee -a "$LOG"
log "smoke spent about \$$(spent)"; pack; stop_pod
