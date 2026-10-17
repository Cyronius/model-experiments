import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    LINEARS, PARENT_DIR, PRISM_DIR, RESULTS, WORK, Weights, absmean_scale, decode_ternary, dequant,
    gptq_ternary, lsq_scale, mse_opt, out_error, rtn, run_info, setup, wikitext_ids, write_json,
)

SITE_OF = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
           "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}
SAVED = ("gptq_prism_s", "gptq_mse_s")


def windows(n, seq, seed):
    tok = AutoTokenizer.from_pretrained(PARENT_DIR)
    ids = wikitext_ids(tok, "train")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randperm(len(ids) // seq - 1, generator=g)[:n] * seq
    return [ids[s:s + seq] for s in starts.tolist()]


def capture(n_cal, n_eval, seq, seed):
    model = AutoModelForCausalLM.from_pretrained(PARENT_DIR, dtype=torch.float32).eval()
    H, phase, hooks = {}, [0], []
    for i, layer in enumerate(model.model.layers):
        for site, mod in (("attn_in", layer.self_attn.q_proj), ("o_in", layer.self_attn.o_proj),
                          ("mlp_in", layer.mlp.gate_proj), ("down_in", layer.mlp.down_proj)):
            H[(i, site)] = [torch.zeros(mod.in_features, mod.in_features) for _ in range(2)]

            def hook(m, inp, key=(i, site)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key][phase[0]].addmm_(x.T, x)

            hooks.append(mod.register_forward_pre_hook(hook))
    seqs = windows(n_cal + n_eval, seq, seed)
    with torch.inference_mode():
        for j, ids in enumerate(seqs):
            phase[0] = 0 if j < n_cal else 1
            model.model(input_ids=ids[None])
            print(f"capture {j + 1}/{len(seqs)}", flush=True)
    for h in hooks:
        h.remove()
    for v in H.values():
        v[0] /= n_cal * seq
        v[1] /= n_eval * seq
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--kbest", type=int, default=8)
    ap.add_argument("--kbest-layers", default="0,7,14,21,27")
    args = ap.parse_args()
    setup(args.threads)
    t0 = time.time()
    cache = WORK / f"hessians_{args.n_cal}_{args.n_eval}_{args.seq}.pt"
    if cache.exists():
        H = torch.load(cache)
    else:
        H = capture(args.n_cal, args.n_eval, args.seq, seed=0)
        torch.save(H, cache)
    print(f"hessians ready in {time.time() - t0:.0f}s", flush=True)
    parent, prism = Weights(PARENT_DIR), Weights(PRISM_DIR)
    kb_layers = {int(x) for x in args.kbest_layers.split(",") if x}
    n_layers = 1 + max(k[0] for k in H)
    out = {"run": run_info(**vars(args)), "layers": {}}
    prev = RESULTS / "layer_error.json"
    if prev.exists():
        out["layers"] = json.loads(prev.read_text())["layers"]
    for i in range(n_layers):
        if str(i) in out["layers"] and all((WORK / f"{m}_L{i}.pt").exists() for m in SAVED):
            continue
        saved = {m: {} for m in SAVED}
        rows = {}
        for lin in LINEARS:
            name = f"model.layers.{i}.{'self_attn' if lin in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'}.{lin}.weight"
            W = parent.get(name).float()
            t_p, s_p, _ = decode_ternary(prism.get(name))
            Hc, He = H[(i, SITE_OF[lin])]
            t_mse, s_mse = mse_opt(W)
            s_am = absmean_scale(W)
            codes = {
                "prism": (t_p, s_p),
                "prism_codes_lsq_s": (t_p, lsq_scale(W, t_p)),
                "rtn_prism_s": (rtn(W, s_p), s_p),
                "rtn_absmean": (rtn(W, s_am), s_am),
                "mse": (t_mse, s_mse),
                "gptq_prism_s": (gptq_ternary(W, Hc, s_p), s_p),
                "gptq_mse_s": (gptq_ternary(W, Hc, s_mse), s_mse),
            }
            if i in kb_layers:
                codes[f"kbest{args.kbest}_prism_s"] = (gptq_ternary(W, Hc, s_p, k=args.kbest), s_p)
            row = {}
            for m, (t, s) in codes.items():
                Wq = dequant(t, s)
                row[m] = {"err_eval": out_error(W, Wq, He), "err_cal": out_error(W, Wq, Hc),
                          "agree_prism": (t == t_p).float().mean().item(), "zero": (t == 0).float().mean().item()}
            for m in SAVED:
                saved[m][name] = codes[m][0]
            rows[lin] = row
            print(f"L{i} {lin}: " + " ".join(f"{m}={r['err_eval']:.4f}" for m, r in row.items()), flush=True)
        out["layers"][i] = rows
        for m in SAVED:
            torch.save(saved[m], WORK / f"{m}_L{i}.pt")
        write_json(RESULTS / "layer_error.json", out)
    out["seconds"] = time.time() - t0
    write_json(RESULTS / "layer_error.json", out)


if __name__ == "__main__":
    main()
