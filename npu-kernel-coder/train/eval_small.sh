#!/bin/sh
# Convert a small-model checkpoint (HF dir) to GGUF bf16, serve it with llama-server under one dibs
# GPU hold, and run eval_small.py on the given task files.
#   ./eval_small.sh MODEL_DIR TAG TASKS.jsonl... [-- eval_small.py flags]   results: runs/evals/TAG-<split>.json
set -eu
cd "$(dirname "$0")"
MODEL=$1; TAG=$2; shift 2
L=/home/cyrus/code/phlegm/target/npu-dev/nemotronh/llama.cpp-src
BIN=/home/cyrus/.cache/model-experiments/nkc-bakeoff/llama-bin
GG=/home/cyrus/.cache/model-experiments/nkc-train/gguf/$TAG.gguf
mkdir -p "$(dirname "$GG")" runs/evals
[ -f "$GG" ] || .venv/bin/python "$L/convert_hf_to_gguf.py" "$MODEL" --outtype bf16 --outfile "$GG"
FILES=""; while [ $# -gt 0 ] && [ "$1" != "--" ]; do FILES="$FILES $1"; shift; done
[ $# -gt 0 ] && shift
# A free port, so the health check can only reach our own server (8095 once hit another session's).
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
dibs run gpu --label "npu-kernel-coder small-model eval $TAG" --eta 40m -- sh -c "
  LD_LIBRARY_PATH=$BIN $BIN/llama-server -m $GG --host 127.0.0.1 --port $PORT -c 131072 -np 8 -fa on -ngl 99 --no-webui >runs/evals/$TAG-server.log 2>&1 &
  S=\$!; trap 'kill \$S' EXIT
  until curl -sf 127.0.0.1:$PORT/health >/dev/null; do sleep 2; kill -0 \$S || exit 1; done
  sleep 1; kill -0 \$S || exit 1
  for f in $FILES; do
    python3 eval_small.py http://127.0.0.1:$PORT \$f --out runs/evals/$TAG-\$(basename \$f .jsonl).json $*
  done"
