import argparse
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    LINEARS, PARENT_DIR, PRISM_DIR, RESULTS, WORK, Weights, decode_ternary, dequant, mse_opt, rtn,
    run_info, setup, wikitext_ids, write_json,
)


def linear_names(model):
    for i, layer in enumerate(model.model.layers):
        for lin in LINEARS:
            attn = lin in ("q_proj", "k_proj", "v_proj", "o_proj")
            name = f"model.layers.{i}.{'self_attn' if attn else 'mlp'}.{lin}.weight"
            yield i, name, getattr(layer.self_attn if attn else layer.mlp, lin)


def substitute(model, variant, parent, prism):
    cache = {}
    for i, name, mod in linear_names(model):
        t_p, s_p, _ = decode_ternary(prism.get(name))
        if variant == "prism":
            Wq = dequant(t_p, s_p)
        elif variant == "rtn_prism_s":
            Wq = dequant(rtn(parent.get(name).float(), s_p), s_p)
        elif variant == "mse":
            Wq = dequant(*mse_opt(parent.get(name).float()))
        elif variant in ("gptq_prism_s", "gptq_mse_s"):
            if i not in cache:
                cache = {i: torch.load(WORK / f"{variant}_L{i}.pt")}
            t = cache[i][name]
            s = s_p if variant == "gptq_prism_s" else mse_opt(parent.get(name).float())[1]
            Wq = dequant(t, s)
        else:
            raise ValueError(variant)
        mod.weight.data.copy_(Wq)


def nll(model, wins):
    out = []
    with torch.inference_mode():
        for ids in wins:
            logits = model(input_ids=ids[None]).logits[0, :-1].float()
            out.append(F.cross_entropy(logits, ids[1:]).item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--variants", default="prism,rtn_prism_s,mse,gptq_prism_s,gptq_mse_s,parent")
    args = ap.parse_args()
    setup(args.threads)
    tok = AutoTokenizer.from_pretrained(PARENT_DIR)
    ids = wikitext_ids(tok, "test")
    offs = [j * (len(ids) - args.seq) // (args.n - 1) for j in range(args.n)]
    wins = [ids[o:o + args.seq] for o in offs]
    parent, prism = Weights(PARENT_DIR), Weights(PRISM_DIR)
    out = {"run": run_info(**vars(args)), "offsets": offs, "variants": {}}
    model = None
    variants = args.variants.split(",")
    # substituted variants reuse one Prism model, so the parent has to load last
    for v in sorted(variants, key=lambda v: v == "parent"):
        t0 = time.time()
        if v == "parent":
            model = AutoModelForCausalLM.from_pretrained(PARENT_DIR, dtype=torch.float32).eval()
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(PRISM_DIR, dtype=torch.float32).eval()
            substitute(model, v, parent, prism)
        per = nll(model, wins)
        mean = sum(per) / len(per)
        out["variants"][v] = {"nll": mean, "ppl": math.exp(mean), "per_window": per, "seconds": time.time() - t0}
        print(f"{v}: ppl={math.exp(mean):.3f} ({time.time() - t0:.0f}s)", flush=True)
        write_json(RESULTS / "ppl.json", out)


if __name__ == "__main__":
    main()
