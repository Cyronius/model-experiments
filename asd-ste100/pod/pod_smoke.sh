#!/bin/bash
# 200-request probe. Measures real tokens/s and success rate, then prints an estimate
# for the full run. Intended spend: about $1 to $2. Usage: RATE=0.79 BUDGET=2 bash pod_smoke.sh
source "$(dirname "$0")/common.sh"
guard env;  phase_env || fail env
guard model; fetch_model vol "$VOL_REPO" "*Q4_K_M*" ; VOL=$MODEL_PATH
guard sources
python3 -I "$W/repo/asd-ste100/data/fetch_sources.py" --out "$OUT/sources.jsonl" --n 3000 --md-dir "$W/repo" | tee -a "$LOG"
guard server; start_server "$VOL" 8080 || fail server
guard gen
T0=$(date +%s)
gen rewrite --url http://127.0.0.1:8080 --input "$OUT/sources.jsonl" --out "$OUT/smoke-seed.jsonl" --workers "$SLOTS" --limit 200 | tail -1 | tee "$OUT/smoke.json"
T1=$(date +%s)
python3 - <<PY | tee -a "$LOG"
import json
r = json.load(open("$OUT/smoke.json")); secs = max($T1 - $T0, 1)
n = r["ok"] + r["failed"]; per_req = secs / max(n, 1)
tps = r["gen_tokens"] / secs
print(f"smoke: {n} requests in {secs}s = {per_req:.2f}s each, {tps:.0f} generated tok/s, success {r['ok']}/{n}")
for total in (12000, 30000):
    h = total * per_req / 3600
    print(f"  {total} sources: about {h:.1f} h generation, about \${h * $RATE:.2f} at \${$RATE}/h (before messy, hard, judge phases)")
PY
log "smoke spent about \$$(spent)"; pack; stop_pod
