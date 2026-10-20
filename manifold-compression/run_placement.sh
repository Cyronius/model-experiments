#!/bin/sh
# Collect MLP-input states for layers 6 and 15, then run the placement test; safe to rerun after a dibs requeue (2026-10-18).
set -e
cd "$(dirname "$0")"
[ -f work/Qwen3.5-0.8B-mlp/meta.json ] || python -u collect.py --n-fit 100 --sites 13,31 --suffix mlp --bs 4
python -u m_placement.py --work Qwen3.5-0.8B-mlp --layers 6,15 --n-eval 2
