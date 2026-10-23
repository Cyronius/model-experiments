#!/bin/bash
# Runs the 25/50/100% scaling test (patch 4, 20k steps each) on a RunPod GPU, full data first; 2026-10-19.
cd /workspace/sprite/utilities/sprite-train
for f in 1.0 0.5 0.25; do
  d=/workspace/runs/scale-p4-f$f
  mkdir -p $d
  python -W ignore train.py --data /workspace/trainset.npz --out $d --frac $f --patch 4 --steps 20000 --ckpt-every 2000 >> $d/train.log 2>&1
  echo "finished frac $f at $(date -Is)" >> /workspace/progress.log
done
touch /workspace/DONE
