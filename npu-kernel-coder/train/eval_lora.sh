#!/bin/sh
# Eval runs for a LoRA (or the base model) under one dibs GPU hold, same llama-server settings as
# eval/runs/bakeoff-q36m-fa-mtp3 (Q4_K_M, ctx 40960, flash attention, MTP draft 3, temp 0, no thinking).
#   ./eval_lora.sh LORA_GGUF|none TAG:TASKS [TAG:TASKS ...]      e.g. runs/sft1/sft1-lora.gguf sft1-pilot:pilot
set -u
cd "$(dirname "$0")"
LORA=$1; shift
E=$(cd ../eval && pwd)
BIN=/home/cyrus/.cache/model-experiments/nkc-bakeoff/llama-bin
GGUF=/home/cyrus/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PY=/home/cyrus/code/model-experiments/target/manifold-audit/env/bin/python
PORT=8094
LA=""; [ "$LORA" = none ] || LA="--lora $(realpath "$LORA")"
dibs run gpu --label "npu-kernel-coder eval ($*)" --eta 120m -- sh -c "
  # dibs can grant the GPU before a preemptible trainer has stepped aside; wait for it to exit
  while pgrep -f '^[.]venv/bin/python train[.]py' >/dev/null; do sleep 5; done
  ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
  K=\$!
  for spec in $*; do
    tag=\${spec%%:*}; tasks=\${spec#*:}; out='$E'/runs/\$tag; mkdir -p \$out
    LD_LIBRARY_PATH='$BIN' '$BIN/llama-server' -m '$GGUF' $LA -c 40960 -ngl 99 --jinja -np 1 \
      -fa on --spec-type draft-mtp --spec-draft-n-max 3 --host 127.0.0.1 --port $PORT >> \$out/server.log 2>&1 &
    S=\$!
    t=0
    until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
      sleep 3; t=\$((t+3))
      if ! kill -0 \$S 2>/dev/null || [ \$t -gt 900 ]; then echo \"\$tag: server failed\"; tail -20 \$out/server.log; kill \$S; continue 2; fi
    done
    echo \"\$tag: server up in \${t}s, \$(date +%H:%M)\"
    (cd '$E' && '$PY' run.py --backend llama --model q36m-fa-mtp3+\$tag --tag \$tag --url http://127.0.0.1:$PORT \
      --tasks \$tasks --variant base --rounds 1 --thinking off --max-tokens 8192 --temperature 0)
    kill \$S; wait \$S 2>/dev/null
  done
  kill \$K 2>/dev/null
" < /dev/null 2>&1 | grep --line-buffered -v '^dibs: waiting'
