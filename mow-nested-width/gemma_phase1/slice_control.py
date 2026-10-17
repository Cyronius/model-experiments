"""Control experiment: prefix-slice the FFNs of a Gemma model and measure the
quality curve. Gemma 4 E4B was NOT MatFormer-trained -- prediction: collapse.
Gemma 3n E4B was -- prediction: graceful degradation. The pair of curves is the
'what MatFormer training buys' figure, on production models.

Zeroing gate/up rows >= h is output-identical to true slicing (act(0)*up = 0
contributes nothing through down_proj). Slices run in decreasing order so the
in-place zeroing composes. Absolute wikitext ppl on -it models is not the point;
the relative curve is.
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import Gemma3nForCausalLM
except ImportError:
    Gemma3nForCausalLM = None

repo = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E4B-it"
tok = AutoTokenizer.from_pretrained(repo)
cls = Gemma3nForCausalLM if ("3n" in repo and Gemma3nForCausalLM) else AutoModelForCausalLM
model = cls.from_pretrained(repo, dtype=torch.bfloat16,
                            device_map="cuda").eval()
print("model class:", model.__class__.__name__, flush=True)

# Text-decoder MLPs only: plain nn.Linear projections. The vision/audio towers
# wrap theirs in ClippableLinear and are out of scope.
mlps = [m for n, m in model.named_modules()
        if hasattr(m, "gate_proj") and hasattr(m, "down_proj")
        and isinstance(getattr(m, "gate_proj", None), torch.nn.Linear)]
dims = sorted({m.gate_proj.weight.shape[0] for m in mlps})
print(f"{repo}: {len(mlps)} text MLPs, intermediate dims {dims}", flush=True)

from datasets import load_dataset
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext",
                                           "wikitext-2-raw-v1",
                                           split="test")["text"] if t.strip())
# Gemma is BOS-sensitive: every window gets its own BOS or ppl explodes.
ids = tok(text, add_special_tokens=False,
          return_tensors="pt").input_ids[:, :40960].to("cuda")
bos = torch.tensor([[tok.bos_token_id]], device="cuda")
W = 2048


@torch.no_grad()
def nll():
    tot, n = 0.0, 0
    for i in range(0, ids.shape[1], W):
        w = ids[:, i:i + W]
        if w.shape[1] < 32:
            break
        inp = torch.cat([bos, w], dim=1)
        out = model(inp[:, :-1]).logits.float()
        ls = torch.nn.functional.cross_entropy(
            out.view(-1, out.shape[-1]), inp[:, 1:].reshape(-1))
        tot += ls.item() * w.shape[1]
        n += w.shape[1]
    return tot / n


@torch.no_grad()
def sample():
    msgs = [{"role": "user",
             "content": "What is the capital of France, and name one river there."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=40, do_sample=False)
    npr = enc["input_ids"].shape[1]
    return tok.decode(out[0, npr:],
                      skip_special_tokens=True).replace("\n", " ")[:150]


for frac in (1.0, 0.75, 0.5, 0.25):
    with torch.no_grad():
        for m in mlps:               # per-module h: dims can vary per layer
            h = int(m.gate_proj.weight.shape[0] * frac)
            m.gate_proj.weight[h:] = 0
            m.up_proj.weight[h:] = 0
    t0 = time.time()
    v = nll()
    print(f"frac {frac:4.2f}: nll {v:7.4f}  ppl {torch.tensor(v).exp():10.2f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"   sample: {sample()}", flush=True)
