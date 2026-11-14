#!/bin/sh
# Smoke test for Qwen3.6-35B-A3B runtime options on llama.cpp build 12060 (Vulkan):
# MTP speculative decoding (--spec-type draft-mtp) and quantized KV cache, with
# flash attention off and on. One real pilot prompt per config, 600 output tokens.
# Prints server-side timings (incl. draft acceptance) and whether the text degenerates.
#   dibs run gpu --label ... -- ./mtp_smoke.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-/home/cyrus/.cache/model-experiments/nkc-bakeoff/llama-bin}
GGUF=${GGUF:-/home/cyrus/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
PROMPT=${PROMPT:-$HERE/runs/bakeoff-qwen3.6-35b-a3b/T314bc26723/prompt.txt}
PORT=${PORT:-8092}
OUT=${OUT:-$HERE/runs/mtp-smoke}
mkdir -p "$OUT"
python3 - "$PROMPT" > "$OUT/body.json" <<'EOF'
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": open(sys.argv[1]).read()}], "max_tokens": 600,
                  "temperature": 0, "seed": 20261105, "chat_template_kwargs": {"enable_thinking": False}}))
EOF
run() {
  name=$1; shift
  LD_LIBRARY_PATH="$BIN" "$BIN/llama-server" -m "$GGUF" -c 40960 -ngl 99 --jinja -np 1 "$@" \
    --host 127.0.0.1 --port $PORT > "$OUT/$name.log" 2>&1 &
  S=$!
  t=0
  until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
    sleep 2; t=$((t+2))
    if ! kill -0 $S 2>/dev/null || [ $t -gt 600 ]; then
      echo "$name: server failed: $(grep -iE 'error|fail|unsupported|abort' "$OUT/$name.log" | tail -2)"; kill $S 2>/dev/null; return; fi
  done
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$OUT/body.json" > "$OUT/$name.json"
  python3 - "$name" "$OUT/$name.json" <<'EOF'
import json, re, sys
name, f = sys.argv[1], sys.argv[2]
try:
    j = json.load(open(f))
except Exception as e:
    print(f"{name}: bad response ({e})"); sys.exit()
t = j.get("timings", {}); c = j["choices"][0]["message"].get("content") or ""
degen = bool(re.search(r"(.)\1{100,}", c))
acc = t.get("draft_n_accepted"); dn = t.get("draft_n")
rate = f"{acc}/{dn}={acc/dn:.2f}" if dn else "-"
print(f"{name}: prefill {t.get('prompt_per_second', 0):.0f} tok/s ({t.get('prompt_n')} tok), "
      f"decode {t.get('predicted_per_second', 0):.1f} tok/s ({t.get('predicted_n')} tok), draft {rate}, degenerate={degen}")
print("   ", c[:160].replace("\n", " | "))
EOF
  kill $S; wait $S 2>/dev/null
}
run base          -fa off
run mtp2          -fa off --spec-type draft-mtp --spec-draft-n-max 2
run mtp3          -fa off --spec-type draft-mtp --spec-draft-n-max 3
run mtp2-kvq8     -fa off --spec-type draft-mtp --spec-draft-n-max 2 -ctk q8_0 -ctv q8_0
run mtp2-kvq4     -fa off --spec-type draft-mtp --spec-draft-n-max 2 -ctk q4_0 -ctv q4_0
run mtp2-fa-kvq4  -fa on  --spec-type draft-mtp --spec-draft-n-max 2 -ctk q4_0 -ctv q4_0
run fa-kvq4       -fa on  -ctk q4_0 -ctv q4_0
