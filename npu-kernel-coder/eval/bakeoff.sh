#!/bin/sh
# Base-model bakeoff on the 12 pilot tasks: one dibs GPU hold per model.
#
#   ./bakeoff.sh [tag ...]        (default: every row of bakeoff_models.tsv)
#   TASKS=T17a0bce0c8 ./bakeoff.sh clover-35b-a3b    (other task selection)
#   MODELS=other.tsv ./bakeoff.sh                    (other model table)
#
# Rows of bakeoff_models.tsv: tag, gguf, ctx, thinking, max_tokens,
# temperature, and optionally variant (base | ex1: one worked example from
# the same design, 6 of the 12 pilot tasks; rN: up to N attempts with the
# toolchain's feedback, see run.py --rounds) and llama-server flags (default
# "-fa off"). A row with a variant is selected as <tag>-<variant>.
# Each model is served by llama.cpp build 12060 (Vulkan; has flint, clover_moe,
# qwen35, qwen35moe, qwen3moe and Q2_0), copied to $BIN so other sessions'
# rebuilds can't change it mid-run. Answers go to runs/bakeoff-<tag>[-ex1]/
# and are scored like every other run; server logs go next to them.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-/home/cyrus/.cache/model-experiments/nkc-bakeoff/llama-bin}
PY=${PY:-$HERE/../../target/manifold-audit/env/bin/python}
PORT=${PORT:-8091}
TASKS=${TASKS:-pilot}
MODELS=${MODELS:-$HERE/bakeoff_models.tsv}
want="$*"
grep -v '^#' "$MODELS" | while IFS="$(printf '\t')" read tag gguf ctx thinking maxtok temp variant sargs; do
  [ -z "$tag" ] && continue
  variant=${variant:-base}
  rounds=1; pvar=$variant
  case $variant in r[0-9]*) rounds=${variant#r}; pvar=base;; esac
  sargs=${sargs:--fa off}
  sel=$tag; [ "$variant" != base ] && sel="$tag-$variant"
  if [ -n "$want" ] && ! echo " $want " | grep -q " $sel "; then continue; fi
  if [ ! -f "$gguf" ]; then echo "skip $tag: $gguf missing"; continue; fi
  name="bakeoff-$tag"
  [ "$variant" != base ] && name="$name-$variant"
  out="$HERE/runs/$name"
  mkdir -p "$out"
  expect=12; [ "$pvar" = ex1 ] && expect=6; [ "$TASKS" != pilot ] && expect=999
  if [ "$(ls "$out"/*/result.json 2>/dev/null | wc -l)" -ge $expect ]; then echo "done already: $name"; continue; fi
  echo "== $name ($(date +%H:%M))"
  dibs wait --pressure >/dev/null 2>&1  # don't load a 20 GB model while the machine is short of RAM or CPU
  dibs run gpu --label "npu-kernel-coder bakeoff $name" --eta 60m -- sh -c "
    dibs extend gpu --eta 60m >/dev/null 2>&1  # the ETA above counts from queueing, not from getting the GPU
    # Keep renewing: on 2026-11-07 a 27B row outlived its ETA and dibs gave the GPU to a ROCm fine-tune mid-run.
    ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &  # detached from the log pipe, or its sleep holds the pipe open after the row
    K=\$!
    LD_LIBRARY_PATH='$BIN' '$BIN/llama-server' -m '$gguf' -c $ctx -ngl 99 --jinja -np 1 $sargs \
      --host 127.0.0.1 --port $PORT >> '$out/server.log' 2>&1 &
    S=\$!
    t=0
    until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
      sleep 3; t=\$((t+3))
      if ! kill -0 \$S 2>/dev/null || [ \$t -gt 900 ]; then echo 'server failed to start'; tail -5 '$out/server.log'; kill \$S \$K 2>/dev/null; exit 1; fi
    done
    echo \"loaded in \${t}s\"
    cd '$HERE' && '$PY' run.py --backend llama --model '$tag' --tag 'bakeoff-$tag' --url http://127.0.0.1:$PORT \
      --tasks $TASKS --variant $pvar --rounds $rounds --thinking $thinking --max-tokens $maxtok --temperature $temp
    kill \$S \$K 2>/dev/null; wait \$S 2>/dev/null
  " < /dev/null 2>&1 | grep --line-buffered -v '^dibs: waiting'
done
