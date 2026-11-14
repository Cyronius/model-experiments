#!/usr/bin/env bash
# Equal-budget float-only control for the original 150M-token 2a_384 run.
set -euo pipefail
cd "$(dirname "$0")"
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p logs runs results
trap 'echo failed > control.status' ERR
echo setup > control.status
python -m pip install -q 'numpy<3' 'mteb==1.39.7' 'transformers==4.57.6' sentence-transformers datasets pillow scipy
python -c 'import torch; assert torch.cuda.is_available(); x=torch.randn(512,512,device="cuda"); print("CUDA",(x@x).norm().item(),torch.cuda.get_device_name())'
echo corpus > control.status
if [ ! -f work/stats.json ]; then python data.py prep --out work > logs/corpus.log 2>&1; fi
python - <<'PY'
import json
s=json.load(open('work/stats.json'))
assert (s['texts'],s['pairs'],s['tokens']) == (4252720,2337599,145428260),s
print('corpus matches Grid 1/2:',s)
PY
echo teacher > control.status
# /workspace is NFS. Random memmap reads/writes there stalled the GPU; keep
# derived arrays on container-local storage without changing numerical settings.
control_work=/tmp/tern-embed-float-control-work
mkdir -p "$control_work"
for file in ids.npy offsets.npy held.npy pairs.npy stsb_dev.json stats.json; do
  cp "work/$file" "$control_work/$file"
done
python data.py teacher --work "$control_work" --batch 512 > logs/teacher.log 2>&1
echo smoke > control.status
python train.py --name float_smoke --init TaylorAI/bge-micro-v2 --layer-ids 0,2 \
  --work "$control_work" --float-frac 1 --max-steps 4 --batch 16 --eval-every 2 --runs runs_smoke > logs/smoke.log 2>&1
python - <<'PY'
from pathlib import Path
from model import Student
m=Student.load(Path('runs_smoke/float_smoke/model.pt'))
assert not m.quantized
assert m.dims['d']==384 and m.dims['layers']==2
print('float checkpoint smoke passed')
PY
echo train > control.status
python train.py --name float150 --init TaylorAI/bge-micro-v2 --layer-ids 0,2 \
  --work "$control_work" --float-frac 1 --tokens 150000000 --seed 0 > logs/train.log 2>&1
echo evaluate > control.status
python evaluate.py run float150=runs/float150/model.pt --results results \
  --tasks STSBenchmark SciFact NFCorpus > logs/evaluate.log 2>&1
cp "$control_work/teacher.npy" work/teacher.npy
python -m pip freeze > environment.txt
echo done > control.status
