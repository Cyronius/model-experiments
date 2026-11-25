#!/bin/bash
# Pedal time-step grid 2 (edge-aware): asinh-compressed target (S = 10 mV, Huber 1 in that space),
# 30% of each batch from rows where some node jumps > 100 mV, halting prior weight 0.001 (was 0.01).
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/analog-step2
run() {  # name eta args...
  name=$1; eta=$2; shift 2
  arbiter run gpu --label "adaptive-loop pedal step2 $name" --eta $eta -- bash -c "
    DATA_DIR=analog-step HALT_RULE=cdf $PY train_analog.py $* --steps 12000 --kl 0.001 --asinh 0.01 --huber 1.0 \
      --edge-frac 0.3 --name $name --out results/analog-step2 > results/analog-step2/$name.log 2>&1 &&
    OUT_DIR=analog-step2 HALT_RULE=cdf $PY eval_rollout.py --name $name $* --asinh 0.01 > results/analog-step2/$name-rollout.log 2>&1"
}
run step2-loop 30m --kind oploop --K 1
run step2-oploop-k8 85m --kind oploop --K 8
