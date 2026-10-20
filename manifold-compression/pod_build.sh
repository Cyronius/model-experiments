#!/bin/bash
# After the measurement suite finishes, build two smaller Qwen3.5-2B chat models (MLPs at 75% and 50% of their neurons) (2026-10-18).
W=/workspace
cd "$W/model-experiments/manifold-compression"
SES="--model Qwen/Qwen3.5-2B --n-fit 64 --n-eval 8 --n-eval-copy 64 --rich --stride 8 --bs 8"
while [ ! -f "$W/DONE.flag" ]; do [ -f "$W/FAILED.flag" ] && exit 1; sleep 20; done
python -u build_pruned.py $SES --chat --keep 0.75 --out "$W/pruned-2b-chat-mlp75" > "$W/logs/build75.log" 2>&1
python -u build_pruned.py $SES --chat --keep 0.5 --out "$W/pruned-2b-chat-mlp50" > "$W/logs/build50.log" 2>&1
touch "$W/BUILT.flag"
