#!/bin/bash
# Diagnosis: can a loop memorise 1k pedal circuits? No step cost, eval on the same 1k.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "of-huber005|--huber 0.05" "of-huber1|--huber 1.0"; do
  name=${cfg%%|*}; a=${cfg#*|}
  TRAIN_SET=train1k TEST_SET=train1k $PY train_analog.py --kind oploop --K 1 --steps 6000 --beta-start 2 \
    --freeze-frac 0 --eval-every 2000 $a --name $name --out results/analog-diag > results/analog-diag/$name.log 2>&1
done
