#!/bin/bash
# Pedal time-step grid 1: predict each node's voltage change over one 48 kHz sample, then run closed
# loop for 480 samples. Halting prior from step 0, cdf rule, Huber 0.05 + MSE. Waits for the data.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
mkdir -p results/analog-step
while [ ! -f cache/analog-step/train.npz ] || pgrep -f "[g]en_analog_step" >/dev/null; do sleep 60; done
run() {  # name eta args...
  name=$1; eta=$2; shift 2
  arbiter run gpu --label "adaptive-loop pedal step $name" --eta $eta -- bash -c "
    DATA_DIR=analog-step HALT_RULE=cdf $PY train_analog.py $* --steps 12000 --kl 0.01 --mse 1 --name $name \
      --out results/analog-step > results/analog-step/$name.log 2>&1 &&
    HALT_RULE=cdf $PY eval_rollout.py --name $name $* > results/analog-step/$name-rollout.log 2>&1"
}
run step-oploop-k8 85m --kind oploop --K 8
run step-loop 30m --kind oploop --K 1
run step-loop-h2048 50m --kind oploop --K 1 --hidden 2048
