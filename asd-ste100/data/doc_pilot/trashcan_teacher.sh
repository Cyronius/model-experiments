#!/usr/bin/env bash
# Teacher server on the "trashcan" Mac Pro (Xeon E5-2697 v2, 2x FirePro D700 6 GB, Vulkan/RADV),
# for pipeline_v5.py with TEACHER_URL=http://100.87.93.68:8090/v1/chat/completions.
# Ternary Bonsai 2 27B (a ternary Qwen3.8-27B) on both GPUs, with the owner's PrismML llama.cpp
# build in ~/Work/bonsai-d700 (used read-only). It writes and checks: on this hardware a
# judge of another family does not fit next to it (see data/doc_pilot/README.md, judge table).
# Bound to the tailscale address only.
#   copy:   tailscale ssh cyrus@trashcan 'mkdir -p ~/Work/ste-teacher' && \
#           tailscale ssh cyrus@trashcan 'cat > ~/Work/ste-teacher/trashcan_teacher.sh' < trashcan_teacher.sh
#   start:  systemd-run --user --unit ste-teacher bash ~/Work/ste-teacher/trashcan_teacher.sh
#   stop:   systemctl --user stop ste-teacher       logs: journalctl --user -u ste-teacher
# The unit lives in the owner's user session: it stops if they log out of the desktop.
set -euo pipefail
B=$HOME/Work/bonsai-d700
exec "$B/llama.cpp/build/bin/llama-server" -m "$B/models/Ternary-Bonsai-2-27B-Q2_0.gguf" \
    --device Vulkan0,Vulkan1 --split-mode layer --tensor-split 1,1 -ngl 99 \
    -t 12 -b 128 -ub 64 -fa off -c 32768 -np 4 --cache-ram 0 \
    --alias bonsai2-27b --host "${HOST:-100.87.93.68}" --port "${PORT:-8090}"
