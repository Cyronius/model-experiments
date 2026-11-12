#!/bin/bash
# Runs on the rented pod (pod_small.py uploads it with train_small.py, memguard.py and the task data).
# Stages go to outputs/status.json; pod_small.py syncs outputs/ back and deletes the pod at the end.
#   TIME_BUDGET_H: training hours (the schedule finishes inside it)
set -uo pipefail
cd /workspace/nkc
O=outputs
mkdir -p $O
stage() { printf '{"stage": "%s", "time": %s}\n' "$1" "$(date +%s)" > $O/status.json; echo "stage $1 $(date -u +%FT%TZ)" >> $O/worker.log; }
fail() { echo "FAILED: $1" >> $O/worker.log; stage failed; exit 1; }

stage setup
python -m venv --system-site-packages env > $O/venv.log 2>&1 || fail venv
env/bin/pip install --no-cache-dir transformers==5.14.1 tokenizers==0.22.2 huggingface-hub==1.33.0 \
  safetensors==0.8.0 fla-core==0.6.0 flash-linear-attention==0.6.0 einops > $O/setup.log 2>&1 || fail pip
nvidia-smi > $O/gpu.txt 2>&1
env/bin/python -c "import torch, fla, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" \
  >> $O/gpu.txt 2>&1 || fail import

stage download
if [ -f init/model.safetensors ]; then
  mv init model  # continue from the uploaded checkpoint
else
  env/bin/hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir model \
    > $O/download.log 2>&1 || fail download
fi
gunzip -f data/train.jsonl.gz data/val.jsonl.gz || fail data

stage train
env/bin/python -u train_small.py model data/train.jsonl data/val.jsonl --out $O/run --require-fla \
  --time-budget-h "$TIME_BUDGET_H" ${TRAIN_ARGS:-} --save-min 1e9 --mem-floor-gb 4 > $O/train.log 2>&1 || fail train
stage complete
