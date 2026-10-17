"""Prefill (compute-bound) timing per rung -- where FFN FLOP savings must show.
Same surgery as ladder_bench; times a 2048-token forward, batch 4."""
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma3nForCausalLM

REPO = "google/gemma-3n-E4B-it"
tok = AutoTokenizer.from_pretrained(REPO)
model = Gemma3nForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16,
                                           device_map="cuda").eval()
mlps = [m for n, m in model.named_modules()
        if hasattr(m, "gate_proj") and isinstance(getattr(m, "gate_proj", None), nn.Linear)]
FULL = mlps[0].gate_proj.weight.shape[0]

def sl(lin, out_keep=None, in_keep=None):
    w = lin.weight
    if out_keep is not None: w = w[:out_keep]
    if in_keep is not None: w = w[:, :in_keep]
    new = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    with torch.no_grad(): new.weight.copy_(w)
    return new

x = torch.randint(3, 200000, (4, 2048), device="cuda")
for frac in (1.0, 0.875, 0.75, 0.625, 0.5):
    h = int(FULL * frac)
    with torch.no_grad():
        for m in mlps:
            if m.gate_proj.weight.shape[0] > h:
                m.gate_proj = sl(m.gate_proj, out_keep=h)
                m.up_proj = sl(m.up_proj, out_keep=h)
                m.down_proj = sl(m.down_proj, in_keep=h)
    torch.cuda.empty_cache()
    with torch.no_grad():
        model(x[:, :64])                      # warm
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3): model(x)
        torch.cuda.synchronize()
    tps = 3 * x.numel() / (time.time() - t0)
    print(f"frac {frac:5.3f}: prefill {tps/1000:7.1f}k tok/s", flush=True)
