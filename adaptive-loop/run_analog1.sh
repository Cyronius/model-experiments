#!/bin/bash
# Analog grid 1, Big Muff-family pedals: op library vs one block vs param-matched block vs dense.
# 12k steps at batch 128, bf16, trained with freezing on half the batches.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/analog
until [ -f cache/analog/train.npz ]; do sleep 30; done
for cfg in "a-oploop-k8|--kind oploop --K 8" "a-loop|--kind oploop --K 1" \
           "a-loop-h2048|--kind oploop --K 1 --hidden 2048" "a-dense|--kind dense"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop pedals $name" --eta 100m -- \
    $PY train_analog.py $a --steps 12000 --name "$name" > "results/analog/$name.log" 2>&1
done
