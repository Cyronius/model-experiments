#!/bin/bash
# Grid 3, circuit task, d=48 controls for oploop-k8-d48-fz:
#   loop-d48-h768-fz : one block with the same parameter count as the 8-expert library (8x active MLP compute)
#   loop-d48-b003-fz : one block, lower step cost, so it can use as many steps as the library did
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "loop-d48-h768-fz|--kind oploop --K 1 --d 48 --hidden 768" \
           "loop-d48-b003-fz|--kind oploop --K 1 --d 48 --hidden 96 --beta 0.003"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop circuit $name" --eta 45m -- \
    $PY train_circuit.py $a --freeze-frac 0.5 --bs 256 --steps 8000 --eval-every 2000 --name "$name" \
    > "results/circuit/$name.log" 2>&1
done
