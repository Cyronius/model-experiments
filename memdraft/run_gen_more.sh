#!/bin/bash
# The 9B (Q4_K_M) writes answers to more coding prompts on 8 CPU threads at low priority: text for
# the reranker and the table to learn from. It starts after the first 2000 prompts of gen_code.py's
# order, the pool code_study.py holds its calibration and test prompts out of, so every row here is
# free to train on. Rows land in target/memdraft/code/rows-9b-more.jsonl as they finish: about
# 100K tokens (190 answers) an hour with 16 answers at a time on a busy machine.
#
#   setsid nohup memdraft/run_gen_more.sh 12000 >> target/memdraft/code/gen-more.log 2>&1 < /dev/null &
#   cancel: pkill -f '[g]en_code.py'     (rows already written stay; starting again continues, once
#           `pgrep -f '[g]en_code.py'` shows nothing: the old one needs up to 20 s to end its server)
set -u
cd "$(dirname "$0")/.."
UPTO=${1:?answer the prompts from place 2000 up to this place in the order}
mkdir -p target/memdraft/code
dibs run cpu port-8455 --label "MemDraft: the 9B writes more code for the reranker (8 CPU threads, low priority)" --eta "${ETA:-55h}" -- \
  nice -n 10 target/manifold-audit/env/bin/python memdraft/gen_code.py --skip 2000 --count "$UPTO" --parallel 16 \
  --out target/memdraft/code/rows-9b-more.jsonl -- -ngl 0 -t "${THREADS:-8}"
rc=$?
echo "$(date +%T) exit $rc"
[ "$rc" -eq 0 ] && touch "target/memdraft/code/gen-more-$UPTO.done"
