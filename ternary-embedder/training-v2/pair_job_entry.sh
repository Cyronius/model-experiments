#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ternary-pilot/ternary-embedder
trap 'printf "failed\n" > training-v2/pair.status' ERR
trap 'printf "failed\n" > training-v2/pair.status; exit 143' TERM INT
export PIP_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
printf 'setup\n' > training-v2/pair.status
python -m pip install -q 'transformers==5.20.2' 'numpy>=1.26'
python - <<'PY'
import torch,json
assert torch.cuda.is_available()
print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'bf16':torch.cuda.is_bf16_supported()}),flush=True)
assert torch.cuda.is_bf16_supported()
PY
printf 'running\n' > training-v2/pair.status
bash "training-v2/${1:-run_data_first.sh}"
python -m pip freeze > training-v2/pair-environment.txt
printf 'done\n' > training-v2/pair.status
