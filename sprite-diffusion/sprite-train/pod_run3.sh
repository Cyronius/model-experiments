#!/bin/bash
# Quality run: patch 2 (256 tokens), category-only, full data; resumes from ckpt.pt; 2026-10-19.
cd /workspace/sprite/utilities/sprite-train
d=/workspace/runs/q-p2-cat
mkdir -p $d
python -W ignore train.py --data /workspace/trainset.npz --out $d --patch 2 --steps ${STEPS:-200000} --ckpt-every 10000 >> $d/train.log 2>&1
echo "finished at $(date -Is)" >> /workspace/progress3.log
touch /workspace/DONE3
