#!/bin/bash
# Head to head on Qwen3.5-9B: the table drafter against llama.cpp's own lookup drafters
# ("prompt lookup"), in tokens per second on the same prompts.
#
#   memdraft/run_lookup.sh vk     # GPU through Vulkan, in turns of at most 9 minutes
#   memdraft/run_lookup.sh cpu    # 8 CPU threads, about 20 minutes
#   memdraft/run_lookup.sh cpu-turns   # the two best CPU arms and plain, taking turns per prompt, 10 minutes
#
# The GPU run takes turns. It asks dibs for the GPU once nothing is on it except jobs that run
# under `dibs run --preemptible` (those hand over when asked), waits inside its hold until the GPU
# is really empty, and then runs plain decoding and as many of the arms still owed as fit in
# LIMIT seconds (520: a 10-minute break between another session's jobs, with a margin). It stops
# its server within seconds if another process shows up on the GPU. Arms left over wait GAP
# seconds (900) and go on the next turn, with plain decoding again as their baseline, so every
# turn writes its own results file. Other work on the machine still runs beside it, so these are
# pilot numbers; run_measure.sh is the clean version.
#
#   setsid nohup memdraft/run_lookup.sh vk > target/memdraft/lookup-vk.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_lookup.sh'; pkill -f '[l]ive_bench.py.*lookup'
set -u
cd "$(dirname "$0")/.."
BACKEND=${1:?vk, cpu or cpu-turns}
PY=target/manifold-audit/env/bin/python
S=$HOME/.cache/model-experiments/memdraft
T=$PWD/target/memdraft/9b-all-c3.mdt
OUT=memdraft/results/lookup-9b-$BACKEND-$(date +%Y%m%d-%H%M)
TABLE="--spec-table-file $T"
# The stock arms run as llama.cpp ships them. On Qwen3.5 that means a checkpoint before every
# drafted step and a second decode after a rejection; the table arms use per-token snapshots.
STOCK=(--arm "ngram-simple=--spec-type ngram-simple" --arm "ngram-mod=--spec-type ngram-mod" --arm "ngram-cache=--spec-type ngram-cache")

case $BACKEND in
  vk)
    # Fixed two-token drafts: the batch shape never changes, so the graph is never rebuilt.
    MINE="$TABLE --spec-table-n-max 2 --spec-table-pad"
    declare -A ARM=(
      [table]="--spec-type ngram-table $MINE"
      [ngram-cache]="--spec-type ngram-cache"
      [ngram-simple]="--spec-type ngram-simple"
      [ngram-mod]="--spec-type ngram-mod"
      [simple+table]="--spec-type ngram-simple,ngram-table $MINE"
      [table-request-only]="--spec-type ngram-table $MINE --spec-table-source context"
    )
    LEFT=(table ngram-cache ngram-simple ngram-mod simple+table table-request-only)   # most wanted first
    turn=0
    while [ ${#LEFT[@]} -gt 0 ] && [ $turn -lt "${TURNS:-8}" ]; do
      turn=$((turn + 1))
      # Look at the GPU itself before asking dibs: a hold taken while a job that cannot be
      # preempted is still running would sit on the GPU record and keep that session's next job out.
      $PY memdraft/gpu_alone.py --wait --preemptible-ok --checks 2 --gap 5
      ARMS=(--arm none=)
      for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
      dibs run gpu port-8455 --label "MemDraft against prompt lookup (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
        $PY memdraft/live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-520}" --arm-seconds 75 \
        --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
        --prompts memdraft/prompts/live-v1.jsonl --per-domain 2 --max-tokens 160 --out "$OUT-turn$turn.json" -- -ngl 99 -c 8192 \
        "${ARMS[@]}"
      rc=$?
      echo "$(date +%T) turn $turn: exit $rc"
      LEFT=($($PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$OUT-turn$turn.json" "${LEFT[@]}"))
      touch "$OUT-turn$turn.done"
      echo "$(date +%T) turn $turn over; still to run: ${LEFT[*]:-nothing}"
      [ ${#LEFT[@]} -gt 0 ] && sleep "${GAP:-900}"
    done
    ;;
  cpu)
    # On CPU one more token in the batch costs most of a step, so draft only when nearly sure.
    SURE="$TABLE --spec-table-n-max 2 --spec-table-tau 0.8"
    dibs run cpu port-8455 --label "MemDraft against prompt lookup (9B on 8 CPU threads, 20 min)" --eta 25m -- \
      $PY memdraft/live_bench.py \
      --server "$S/llama.cpp/build-cpu/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
      --prompts memdraft/prompts/live-v1.jsonl --per-domain 2 --max-tokens 128 --out "$OUT.json" -- -ngl 0 -t 8 -c 8192 \
      --arm none= \
      --arm "table-sure=--spec-type ngram-table $SURE" \
      --arm "table-pad2=--spec-type ngram-table $TABLE --spec-table-n-max 2 --spec-table-pad" \
      "${STOCK[@]}" \
      --arm "simple+table-sure=--spec-type ngram-simple,ngram-table $SURE"
    ;;
  cpu-turns)
    # CPU speed drifts within minutes on this machine (the first cpu run had plain decoding anywhere
    # from 4.5 to 9.7 tok/s), so here all arms answer one prompt back to back before the next prompt.
    dibs run cpu port-8455 --label "MemDraft against prompt lookup (9B on 8 CPU threads, arms in turns, 10 min)" --eta 12m -- \
      $PY memdraft/live_bench.py --chunk 1 \
      --server "$S/llama.cpp/build-cpu/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
      --prompts memdraft/prompts/live-v1.jsonl --per-domain 2 --max-tokens 96 --out "$OUT.json" -- -ngl 0 -t 8 -c 8192 \
      --arm none= \
      --arm "table-pad2=--spec-type ngram-table $TABLE --spec-table-n-max 2 --spec-table-pad" \
      --arm "ngram-cache=--spec-type ngram-cache"
    ;;
  *) echo "unknown backend $BACKEND"; exit 2;;
esac
echo "exit $?"
touch "$OUT.done"
