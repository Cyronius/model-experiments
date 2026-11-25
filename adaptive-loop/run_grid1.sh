#!/bin/bash
# Grid 1, circuit task: dense vs looped block vs op library, 8k steps at batch 256, bf16.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/circuit
for cfg in "oploop-k8|--kind oploop --K 8" "loop|--kind oploop --K 1" "dense|--kind dense"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop circuit $name" --eta 40m -- \
    $PY train_circuit.py $a --bs 256 --steps 8000 --eval-every 2000 --name "$name" > "results/circuit/$name.log" 2>&1
done
