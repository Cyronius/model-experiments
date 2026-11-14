"""Is the 4-5% logits gap between causal_sdpa (flash kernel) and transformers' sdpa path a bug or bf16 noise?

    dibs run gpu -- PYTHONPATH=.fla .venv/bin/python attn_probe.py Q4_MODEL_DIR data/val.jsonl

Runs the same input through the model under three attention implementations (causal_sdpa, sdpa,
eager) and compares each pair: logits relative difference, next-token argmax agreement, and the
answer-token loss of a real validation example cut to --tokens. If eager vs sdpa (two exact
implementations of the same math) differs by about as much as causal_sdpa vs sdpa, the gap is
rounding amplified by 40 layers of top-8 routing, not a wrong kernel.
"""
import argparse
import itertools
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("val", type=Path)
    ap.add_argument("--tokens", type=int, default=2048)
    a = ap.parse_args()
    import memguard
    memguard.start(12)
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    import q4moe

    tok = AutoTokenizer.from_pretrained(a.model)
    e = json.loads(a.val.open().readline())
    ids = tok(tok.apply_chat_template(e["messages"], tokenize=False, enable_thinking=False),
              add_special_tokens=False)["input_ids"][-a.tokens:]
    x = torch.tensor([ids], device="cuda")
    model = q4moe.load(a.model)
    out = {}
    with torch.no_grad():
        for impl in ("causal_sdpa", "sdpa", "eager"):
            model.set_attn_implementation(impl)
            logits = model(input_ids=x).logits[0].float()
            loss = F.cross_entropy(logits[:-1], x[0, 1:]).item()
            out[impl] = (logits, loss)
            print(f"{impl:12s} loss {loss:.4f}", flush=True)
        for p, q in itertools.combinations(out, 2):
            lp, lq = out[p][0], out[q][0]
            rel = ((lp - lq).norm() / lq.norm()).item()
            agree = (lp.argmax(-1) == lq.argmax(-1)).float().mean().item()
            print(f"{p} vs {q}: logits rel diff {rel:.4f}, argmax agreement {agree:.4f}", flush=True)


if __name__ == "__main__":
    main()
