#!/bin/bash
# OCR grid 1: op library vs one block vs param-matched block vs dense. 12k steps at batch 128, bf16,
# trained with freezing on half the batches. Starts after the pedal grid.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/ocr
while pgrep -f "[r]un_analog1" >/dev/null; do sleep 60; done
for cfg in "o-oploop-k8|--kind oploop --K 8" "o-loop|--kind oploop --K 1" \
           "o-loop-h2048|--kind oploop --K 1 --hidden 2048" "o-dense|--kind dense"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop OCR $name" --eta 90m -- \
    $PY train_ocr.py $a --name "$name" > "results/ocr/$name.log" 2>&1
done
