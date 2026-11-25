#!/bin/bash
# OCR grid 3: seeds 1 and 2 of grid 2 (halting prior), each re-evaluated with the cdf halting rule.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for seed in 1 2; do
  for cfg in "o2-oploop-k8|--K 8" "o2-loop|--K 1"; do
    base=${cfg%%|*}; a=${cfg#*|}; name=$base-s$seed
    arbiter run gpu --label "adaptive-loop OCR $name" --eta 75m -- bash -c "
      $PY train_ocr.py --kind oploop $a --kl 0.01 --seed $seed --name $name > results/ocr/$name.log 2>&1 &&
      HALT_RULE=cdf $PY train_ocr.py --kind oploop $a --steps 0 --load results/ocr/$name.pt --name $name-cdf \
        --out results/ocr-cdf > results/ocr/$name-cdf.log 2>&1"
  done
done
