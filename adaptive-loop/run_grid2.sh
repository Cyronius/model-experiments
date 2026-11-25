#!/bin/bash
# Grid 2, circuit task: train with halted tokens frozen on half the batches; routing ablation;
# narrow width (d=48) where one block may run out of capacity. 8k steps at batch 256, bf16.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/circuit
for cfg in "oploop-k8-fz|--kind oploop --K 8" "loop-fz|--kind oploop --K 1" \
           "oploop-k8-d48-fz|--kind oploop --K 8 --d 48 --hidden 96" "loop-d48-fz|--kind oploop --K 1 --d 48 --hidden 96"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop circuit $name" --eta 90m -- \
    $PY train_circuit.py $a --freeze-frac 0.5 --bs 256 --steps 8000 --eval-every 2000 --name "$name" \
    > "results/circuit/$name.log" 2>&1
done
