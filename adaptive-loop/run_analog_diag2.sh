#!/bin/bash
# Diagnosis 2: memorise 1k pedal circuits with an fp32 head, then + Fourier inputs, then + MSE term.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "of2-fp32head|--fourier 0" "of2-fourier|--fourier 6" "of2-fourier-mse|--fourier 6 --mse 1"; do
  name=${cfg%%|*}; a=${cfg#*|}
  TRAIN_SET=train1k TEST_SET=train1k $PY train_analog.py --kind oploop --K 1 --steps 6000 --beta-start 2 \
    --freeze-frac 0 --eval-every 2000 $a --name $name --out results/analog-diag > results/analog-diag/$name.log 2>&1
done
