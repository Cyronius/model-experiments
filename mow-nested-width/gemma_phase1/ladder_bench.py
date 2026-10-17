"""Phase 1 ladder: true sliced submodels of Gemma 3n E4B at 5 FFN widths.
Per rung: wikitext ppl (BOS-per-window), HellaSwag-200 accuracy by option
likelihood, batch-1 decode tokens/sec, and resident VRAM.

Slicing is real shape surgery (smaller nn.Linear modules), so the latency and
memory numbers are what deployment would see. Rungs descend, and prefixes nest,
so each rung slices from the previous one. Rung 1.00 must reproduce yesterday's
zeroing-method ppl (34.46) and rung 0.50 its 44.51 -- built-in consistency check.
"""
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma3nForCausalLM

REPO = "google/gemma-3n-E4B-it"
RUNGS = (1.0, 0.875, 0.75, 0.625, 0.5)

tok = AutoTokenizer.from_pretrained(REPO)
model = Gemma3nForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16,
                                           device_map="cuda").eval()
mlps = [m for n, m in model.named_modules()
        if hasattr(m, "gate_proj") and hasattr(m, "down_proj")
        and isinstance(getattr(m, "gate_proj", None), nn.Linear)]
FULL = mlps[0].gate_proj.weight.shape[0]
print(f"{len(mlps)} text MLPs, full intermediate {FULL}", flush=True)

from datasets import load_dataset
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext",
                                           "wikitext-2-raw-v1",
                                           split="test")["text"] if t.strip())
ids = tok(text, add_special_tokens=False,
          return_tensors="pt").input_ids[:, :40960].to("cuda")
bos_id = tok.bos_token_id
bos = torch.tensor([[bos_id]], device="cuda")
W = 2048

hs = load_dataset("Rowan/hellaswag", split="validation").select(range(200))


def slice_linear(lin: nn.Linear, out_keep=None, in_keep=None) -> nn.Linear:
    w = lin.weight
    if out_keep is not None:
        w = w[:out_keep]
    if in_keep is not None:
        w = w[:, :in_keep]
    new = nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None,
                    device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(w)
        if lin.bias is not None:
            new.bias.copy_(lin.bias[:out_keep] if out_keep is not None else lin.bias)
    return new


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
def hellaswag_acc():
    ok = 0
    for ex in hs:
        ctx = ex["ctx"]
        scores = []
        for end in ex["endings"]:
            e = tok(ctx + " " + end, add_special_tokens=False,
                    return_tensors="pt").input_ids.to("cuda")
            nctx = tok(ctx, add_special_tokens=False,
                       return_tensors="pt").input_ids.shape[1]
            inp = torch.cat([bos, e], dim=1)
            out = model(inp[:, :-1]).logits.float()
            tgt = inp[:, 1:]
            ls = torch.nn.functional.cross_entropy(
                out.view(-1, out.shape[-1]), tgt.reshape(-1), reduction="none")
            scores.append(ls[nctx:].mean().item())   # mean NLL of the ending
        ok += int(min(range(4), key=lambda i: scores[i]) == int(ex["label"]))
    return ok / len(hs)


@torch.no_grad()
def decode_tps(n_new=128, reps=3):
    msgs = [{"role": "user", "content": "Write a short paragraph about rivers."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to("cuda")
    best = 0.0
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        out = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                             min_new_tokens=n_new)
        torch.cuda.synchronize()
        best = max(best, (out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t0))
    return best


print(f"{'frac':>6} {'inter':>7} {'ppl':>8} {'hellaswag':>10} {'tok/s':>7} {'vram GB':>8}",
      flush=True)
for frac in RUNGS:
    h = int(FULL * frac)
    with torch.no_grad():
        for m in mlps:
            if m.gate_proj.weight.shape[0] > h:
                m.gate_proj = slice_linear(m.gate_proj, out_keep=h)
                m.up_proj = slice_linear(m.up_proj, out_keep=h)
                m.down_proj = slice_linear(m.down_proj, in_keep=h)
    torch.cuda.empty_cache()
    v = nll()
    acc = hellaswag_acc()
    tps = decode_tps()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"{frac:6.3f} {h:7d} {torch.tensor(v).exp():8.2f} {acc:10.3f} "
          f"{tps:7.1f} {mem:8.2f}", flush=True)
