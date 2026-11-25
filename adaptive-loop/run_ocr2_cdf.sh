#!/bin/bash
# Re-evaluate OCR grid 2 checkpoints with the halting rule that matches the prior: stop when P(halted) >= 0.5.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "o2-oploop-k8|--K 8" "o2-loop|--K 1"; do
  name=${cfg%%|*}; a=${cfg#*|}
  HALT_RULE=cdf $PY train_ocr.py --kind oploop $a --steps 0 --load results/ocr/$name.pt --name $name-cdf \
    --out results/ocr-cdf > results/ocr/$name-cdf.log 2>&1
done
