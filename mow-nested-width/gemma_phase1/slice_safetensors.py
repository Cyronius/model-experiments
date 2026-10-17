"""Slice Gemma 3n E4B FFNs to 8192 by editing safetensors shards directly.
No transformers import -- matches tensors by shape, so the multimodal towers
pass through untouched and the checkpoint layout stays pristine."""
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SNAP = "/root/.cache/huggingface/hub/models--google--gemma-3n-E4B-it/snapshots/c1221e9c62e34a43ab7ffacd1be0ea71f126ef10"
DST = "/work/hf_half"
H = 8192

os.makedirs(DST, exist_ok=True)
total, sliced = 0, 0
for f in sorted(os.listdir(SNAP)):
    src = os.path.join(SNAP, f)
    dst = os.path.join(DST, f)
    if f.endswith(".safetensors"):
        sd = load_file(src)
        out = {}
        for k, v in sd.items():
            if k.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")) \
                    and tuple(v.shape) == (16384, 2048):
                v = v[:H].contiguous(); sliced += 1
            elif k.endswith("mlp.down_proj.weight") \
                    and tuple(v.shape) == (2048, 16384):
                v = v[:, :H].contiguous(); sliced += 1
            out[k] = v
            total += v.numel() * v.element_size()
        save_file(out, dst, metadata={"format": "pt"})
        print("shard", f, "done", flush=True)
    elif os.path.isfile(src):
        shutil.copy(src, dst)

cfgp = os.path.join(DST, "config.json")
cfg = json.load(open(cfgp))
cfg["text_config"]["intermediate_size"] = H
json.dump(cfg, open(cfgp, "w"), indent=2)

idxp = os.path.join(DST, "model.safetensors.index.json")
if os.path.exists(idxp):
    idx = json.load(open(idxp))
    idx.setdefault("metadata", {})["total_size"] = total
    json.dump(idx, open(idxp, "w"))
print(f"sliced {sliced} tensors (expect 105 = 35 layers x 3), total {total/1e9:.2f} GB")
