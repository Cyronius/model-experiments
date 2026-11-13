#!/bin/bash
# The 9B writes answers to coding prompts on 8 CPU threads at low priority, for code_study.py.
# 22 to 30 tokens per second with 16 answers at a time on a busy machine: 800 prompts (433K tokens)
# took about 4.5 hours. Rows land in target/memdraft/code/rows-9b.jsonl as they finish.
#
#   setsid nohup memdraft/run_gen_code.sh 800 > target/memdraft/code/gen.log 2>&1 < /dev/null &
#   cancel: pkill -f '[g]en_code.py'     (rows already written stay; starting again continues)
set -u
cd "$(dirname "$0")/.."
COUNT=${1:?how many prompts}
mkdir -p target/memdraft/code
dibs run cpu port-8455 --label "MemDraft: the 9B writes code for the code study (8 CPU threads, low priority)" --eta "${ETA:-7h}" -- \
  nice -n 10 target/manifold-audit/env/bin/python memdraft/gen_code.py --count "$COUNT" --parallel 16 \
  --out target/memdraft/code/rows-9b.jsonl -- -ngl 0 -t 8
rc=$?
echo "$(date +%T) exit $rc"
[ "$rc" -eq 0 ] && touch "target/memdraft/code/gen-$COUNT.done"
