#!/bin/bash
# Family audio diagnosis: does the GRU learn without conditioning, and with a gentler encoder?
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
for cfg in "au-fam-nocond|--no-cond" "au-fam-enclr|--enc-lr 1e-4"; do
  name=${cfg%%|*}; a=${cfg#*|}
  arbiter run gpu --label "adaptive-loop audio diag $name" --eta 15m -- $PY train_audio.py --mode family --name $name \
    --hidden 64 --steps 1500 --seg 2048 --load-encoder results/analog-dc/dc-oploop-k8.pt $a --out results/audio-diag \
    > results/audio/$name.log 2>&1
done
