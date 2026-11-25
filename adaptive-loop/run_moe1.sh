#!/bin/bash
# Standard-MoE baselines (distinct layers, each with its own 8 experts), same training as the op library
# (halting prior from step 0, cdf rule at eval). Shallow (4 layers) first, then full depth.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
ocr() {  # name T
  arbiter run gpu --label "adaptive-loop OCR $1" --eta $3 -- bash -c "
    $PY train_ocr.py --kind moe --K 8 --T $2 --kl 0.01 --name $1 > results/ocr/$1.log 2>&1 &&
    HALT_RULE=cdf $PY train_ocr.py --kind moe --K 8 --T $2 --steps 0 --load results/ocr/$1.pt --name $1-cdf \
      --out results/ocr-cdf > results/ocr/$1-cdf.log 2>&1"
}
dc() {  # name T
  arbiter run gpu --label "adaptive-loop pedal DC $1" --eta $3 -- bash -c "
    DATA_DIR=analog-dc $PY train_analog.py --kind moe --K 8 --T $2 --steps 12000 --kl 0.01 --mse 1 --name $1 \
      --out results/analog-dc > results/analog-dc/$1.log 2>&1 &&
    DATA_DIR=analog-dc HALT_RULE=cdf $PY train_analog.py --kind moe --K 8 --T $2 --steps 0 --load results/analog-dc/$1.pt \
      --name $1-cdf --out results/analog-dc-cdf > results/analog-dc/$1-cdf.log 2>&1"
}
ocr o2-moe4 4 25m
dc dc-moe4 4 25m
ocr o2-moe16 16 80m
dc dc-moe24 24 85m
