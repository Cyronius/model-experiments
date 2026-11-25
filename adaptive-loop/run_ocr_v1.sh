#!/bin/bash
# OCR cleaner v1: op library (0.65M params) on fineweb-edu v2 data, 40k steps, trained in 3k-step phases
# so the shared GPU is released between phases. freeze_z so infer.py's sparse path is exact.
cd "$(dirname "$0")"
PY=../target/manifold-rocm/env/bin/python
export OCR_DATA=ocr2
name=ocr-v1-k8
mkdir -p results/ocr2
while :; do
  arbiter run gpu --label "adaptive-loop OCR cleaner v1 phase" --eta 20m -- \
    $PY train_ocr.py --kind oploop --K 8 --steps 40000 --phase-steps 3000 --eval-every 3000 --kl 0.01 --freeze-z \
      --name $name --out results/ocr2 >> results/ocr2/$name.log 2>&1
  rc=$?
  [ $rc -eq 3 ] && continue
  break
done
echo "train exit $rc"
[ $rc -eq 0 ] && HALT_RULE=cdf $PY train_ocr.py --kind oploop --K 8 --steps 0 --freeze-z --load results/ocr2/$name.pt \
  --name $name-cdf --out results/ocr2-cdf > results/ocr2/$name-cdf.log 2>&1
