#!/bin/bash
# Pedal DC bias-point grid: op library vs loop vs same-size loop, halting prior from step 0,
# Huber 0.05 + MSE loss, then re-evaluated with the cdf halting rule.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
export DATA_DIR=analog-dc
for cfg in "dc-oploop-k8|--K 8" "dc-loop|--K 1" "dc-loop-h2048|--K 1 --hidden 2048"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop pedal DC $name" --eta 80m -- bash -c "
    $PY train_analog.py --kind oploop $a --steps 12000 --kl 0.01 --mse 1 --name $name --out results/analog-dc \
      > results/analog-dc/$name.log 2>&1 &&
    HALT_RULE=cdf $PY train_analog.py --kind oploop $a --steps 0 --load results/analog-dc/$name.pt --name $name-cdf \
      --out results/analog-dc-cdf > results/analog-dc/$name-cdf.log 2>&1"
done
