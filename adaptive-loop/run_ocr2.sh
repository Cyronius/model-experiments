#!/bin/bash
# OCR grid 2: grid 1's op library never learned to halt (stuck at 16 steps). Add the PonderNet
# geometric-prior KL from step 0 to op library and loop alike.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "o2-oploop-k8|--kind oploop --K 8" "o2-loop|--kind oploop --K 1"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop OCR $name" --eta 75m -- \
    $PY train_ocr.py $a --kl 0.01 --name "$name" > "results/ocr/$name.log" 2>&1
done
