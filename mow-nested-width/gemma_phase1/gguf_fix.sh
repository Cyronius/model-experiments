#!/usr/bin/env bash
# The converter expects the original multimodal checkpoint layout. full.gguf now
# converts straight from the pristine snapshot; half slices inside the original
# Gemma3nForConditionalGeneration class so the saved layout matches.
set -ex
cd /work
rm -f half.gguf
rm -rf hf_half

SNAP=$(python3 - <<PY
from huggingface_hub import hf_hub_download
import os
print(os.path.dirname(hf_hub_download("google/gemma-3n-E4B-it", "config.json")))
PY
)
echo "snapshot: $SNAP"

test -f /work/full.gguf || python3 llama.cpp/convert_hf_to_gguf.py "$SNAP" --outfile /work/full.gguf --outtype bf16 2>&1 | tail -2

python3 - <<PY
import torch, torch.nn as nn
from transformers import Gemma3nForConditionalGeneration as AutoModelForImageTextToText

REPO = "google/gemma-3n-E4B-it"
model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.bfloat16).eval()
print("class:", model.__class__.__name__, flush=True)

# text MLPs identified by exact full-width shape, so no tower ambiguity
tgt = [m for n, m in model.named_modules()
       if hasattr(m, "gate_proj") and isinstance(getattr(m, "gate_proj", None), nn.Linear)
       and m.gate_proj.weight.shape == (16384, 2048)]
print(len(tgt), "text MLPs", flush=True)
assert len(tgt) == 35

H = 8192
def sl(lin, out_keep=None, in_keep=None):
    w = lin.weight
    if out_keep is not None: w = w[:out_keep]
    if in_keep is not None: w = w[:, :in_keep]
    new = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
    with torch.no_grad(): new.weight.copy_(w)
    return new

with torch.no_grad():
    for m in tgt:
        m.gate_proj = sl(m.gate_proj, out_keep=H)
        m.up_proj = sl(m.up_proj, out_keep=H)
        m.down_proj = sl(m.down_proj, in_keep=H)

model.config.get_text_config().intermediate_size = H
model.save_pretrained("/work/hf_half")
for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
          "special_tokens_map.json", "preprocessor_config.json",
          "processor_config.json", "chat_template.jinja",
          "generation_config.json"):
    import shutil, os
    src = os.path.join("$SNAP", f)
    if os.path.exists(src): shutil.copy(src, "/work/hf_half/")
print("half saved", flush=True)
PY

python3 llama.cpp/convert_hf_to_gguf.py /work/hf_half --outfile /work/half.gguf --outtype bf16 2>&1 | tail -2
ls -la /work/*.gguf

cd /work/llama.cpp
./build/bin/llama-bench -m /work/full.gguf -ngl 99 -p 512 -n 128 -r 3 2>&1 | tail -6
./build/bin/llama-bench -m /work/half.gguf -ngl 99 -p 512 -n 128 -r 3 2>&1 | tail -6
echo BENCH_DONE
