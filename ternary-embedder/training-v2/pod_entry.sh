#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ternary-pilot/ternary-embedder
trap 'echo failed > training-v2/pilot.status' ERR
export PIP_BREAK_SYSTEM_PACKAGES=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/workspace/ternary-pilot/hf-cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p training-v2/runs
printf setup > training-v2/pilot.status
python -m pip install -q 'transformers==5.20.2' datasets scipy pyarrow
python - <<'PY'
import torch,json
assert torch.cuda.is_available()
a=torch.randn(512,512,device='cuda');assert torch.isfinite(a@a).all()
print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}),flush=True)
PY
printf running > training-v2/pilot.status
bash "training-v2/${1:-run_pilot.sh}"
python -m pip freeze > training-v2/environment.txt
printf done > training-v2/pilot.status
