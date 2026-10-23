#!/bin/bash
# Two 100k-step runs on the full data: caption-conditioned, then category-only (baseline); 2026-10-19.
cd /workspace/sprite/utilities/sprite-train
for mode in text cat; do
  d=/workspace/runs/q100k-$mode
  mkdir -p $d
  extra=""; [ $mode = text ] && extra="--text"
  python -W ignore train.py --data /workspace/trainset.npz --out $d --patch 4 --steps 100000 --ckpt-every 5000 $extra >> $d/train.log 2>&1
  echo "finished $mode at $(date -Is)" >> /workspace/progress2.log
done
touch /workspace/DONE2
