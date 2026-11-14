#!/bin/bash
# The wider GPU run of the table drafter with the reranker (Qwen3.5-9B, Vulkan), three questions at once:
#   more prompts   48 held-out OSS-Instruct prompts (code-v2.jsonl, 24 per domain; code-v1's 12 among
#                  them), 192 tokens, two rounds
#   longer drafts  the reranker drafting 3, 4 and 5 tokens (pad), and the drafter alone at 3
#   other work     code-edit-v1.jsonl: 6 whole-file edits and 6 long programs, 640 tokens, two rounds
# Turns of at most 9 minutes (12 OSS prompts or 4 edit/write prompts, every arm, plain decoding first),
# each when the GPU is ours alone (gpu_alone.py, --gpu-guard) and the load average is below LOAD (or
# after waiting LOAD_WAIT seconds for it), GAP seconds apart so other sessions get the GPU between.
# An arm cut short goes again in the next turn on the same prompts, with plain decoding again.
# A preemptible job counts as leaving only while dibs lists its hold (then our `dibs run gpu` makes
# it yield); one whose hold has vanished never hears of us, so then the turn waits for it to end.
# Not `dibs run --measure`: that takes the whole machine. show_wide.py sums the turns.
#
#   setsid nohup memdraft/run_wide_live.sh > target/memdraft/code/wide.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_wide_live.sh'; pkill -f '[l]ive_bench.py.*wide-'
set -u
cd "$(dirname "$0")/.."
PY=target/manifold-audit/env/bin/python
S=$HOME/.cache/model-experiments/memdraft
NEW=$PWD/target/memdraft/code/9b-code-pod-c3
STAMP=$(date +%Y%m%d-%H%M)
OUT=memdraft/results/wide-9b-vk-$STAMP
WORK=target/memdraft/code/wide-$STAMP
mkdir -p "$OUT" "$WORK"
T="--spec-type ngram-table --spec-table-pad"
R="--spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
declare -A ARM=(
  [pod-table-3]="$T --spec-table-n-max 3 --spec-table-file $NEW.mdt"
  [reranker-3]="$T --spec-table-n-max 3 $R"
  [reranker-4]="$T --spec-table-n-max 4 $R"
  [reranker-5]="$T --spec-table-n-max 5 $R"
)
ALL=(reranker-4 reranker-3 reranker-5 pod-table-3)

# The prompt subsets: code-v2's first 24 of each domain in four parts of 12, the edit set in three
# parts of 2 edits and 2 programs.
$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
rows = [json.loads(l) for l in open('memdraft/prompts/code-v2.jsonl')]
seen = {}
pick = [r for r in rows if seen.setdefault(r['domain'], []).append(r) or len(seen[r['domain']]) <= 24]
for n in range(4):
    open(f'{work}/oss-{n}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in pick[12 * n:12 * n + 12]))
ed = [json.loads(l) for l in open('memdraft/prompts/code-edit-v1.jsonl')]
edit, write = [r for r in ed if r['domain'] == 'edit'], [r for r in ed if r['domain'] == 'write']
for n in range(3):
    part = edit[2 * n:2 * n + 2] + write[2 * n:2 * n + 2]
    open(f'{work}/edit-{n}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in part))
EOF

JOBS=()
for round in 1 2; do
  JOBS+=(oss-0:$round edit-0:$round oss-1:$round edit-1:$round oss-2:$round edit-2:$round oss-3:$round)
done

turn=0
for job in "${JOBS[@]}"; do
  part=${job%:*}; round=${job#*:}
  case $part in oss-*) TOKENS=192;; *) TOKENS=640;; esac
  LEFT=("${ALL[@]}")
  for attempt in 1 2 3; do
    turn=$((turn + 1))
    [ $turn -gt 1 ] && sleep "${GAP:-300}"
    waited=0
    while awk -v m="${LOAD:-8}" '{exit !($1 >= m)}' /proc/loadavg && [ $waited -lt "${LOAD_WAIT:-2700}" ]; do
      sleep 60; waited=$((waited + 60))
    done
    until $PY memdraft/gpu_alone.py > /dev/null || { dibs status | grep -q '^gpu .*\[preemptible\]' && $PY memdraft/gpu_alone.py --preemptible-ok > /dev/null; }; do
      sleep 60
    done
    name="$part-r$round-a$attempt"
    echo "$(date +%T) turn $turn: $name, load $(cut -d' ' -f1 /proc/loadavg) after waiting ${waited}s, arms ${LEFT[*]}"
    ARMS=(--arm none=)
    for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
    dibs run gpu port-8455 --label "MemDraft: wider table + reranker run, turn $turn of about 14 (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
      $PY memdraft/live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-540}" --arm-seconds 75 \
      --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
      --prompts "$WORK/$part.jsonl" --max-tokens $TOKENS --out "$OUT/$name.json" -- -ngl 99 -c 8192 "${ARMS[@]}"
    echo "$(date +%T) turn $turn: exit $?"
    LEFT=($($PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$OUT/$name.json" "${LEFT[@]}"))
    [ ${#LEFT[@]} -eq 0 ] && break
    echo "$(date +%T) still to run on $part round $round: ${LEFT[*]}"
  done
done
$PY memdraft/show_wide.py "$OUT" > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
