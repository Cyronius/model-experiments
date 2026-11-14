#!/bin/sh
# Serve the coder model (Qwen3.6-35B-A3B) with the tuned runtime settings, under a dibs GPU hold.
#
#   ./serve_coder.sh [extra llama-server flags]      (port 8093; ETA via ETA=2h)
#
# Settings, from runtime_models.tsv on the 12 pilot tasks (2026-11-08; eval/results.md):
#   --spec-type draft-mtp --spec-draft-n-max 3   the model's own MTP head drafts 3 tokens (96% accepted)
#   -fa on                                       flash attention (no degeneration on qwen3.6 here)
# Decode 42 -> 93 tok/s with builds / hunks applied / gap closed within noise of plain decoding.
# KV cache stays f16: a 4-bit cache (-ctk q4_0 -ctv q4_0) measured no faster, since only 1 layer
# in 4 keeps a KV cache. Use it only when a long context needs the memory.
# MTP output is not bit-identical to plain decoding. The GGUF must keep the MTP layer
# (blk.40.nextn.*): UD-Q4_K_M does, UD-Q4_K_XL does not.
set -u
BIN=${BIN:-/home/cyrus/.cache/model-experiments/nkc-bakeoff/llama-bin}
GGUF=${GGUF:-/home/cyrus/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
PORT=${PORT:-8093}
exec dibs run gpu --label "npu-kernel-coder coder model server" --eta "${ETA:-2h}" -- sh -c "
  ( while sleep 1200; do dibs extend gpu --eta 45m; done ) </dev/null >/dev/null 2>&1 &
  K=\$!
  trap 'kill \$K 2>/dev/null' EXIT
  LD_LIBRARY_PATH='$BIN' '$BIN/llama-server' -m '$GGUF' -c 40960 -ngl 99 --jinja -np 1 \
    -fa on --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 127.0.0.1 --port $PORT $*
"
