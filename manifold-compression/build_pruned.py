# Build a smaller, standard-architecture model: shrink every MLP by keeping its most important neurons and re-solving the down projection by least squares; no gradient training (2026-10-18).
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import add_session_args, batches, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=8)
ap.add_argument("--keep", type=float, default=0.5, help="fraction of MLP neurons kept in every layer")
ap.add_argument("--calib-seqs", type=int, default=128)
ap.add_argument("--ridge", type=float, default=1e-3)
ap.add_argument("--out", required=True)
ap.add_argument("--chat", action="store_true", help="sample with the chat template (post-trained models)")
ap.add_argument("--self-chat", type=int, default=0, help="add this many of the model's own chat answers (real template) to the calibration set")
ap.add_argument("--shape", type=float, default=0.0,
                help="0..1: weight neuron choice and the refit by how much the output is sensitive to each token and direction")
ap.add_argument("--tag", default="", help="suffix for the results file name")
args = ap.parse_args()

S = session_from(args)
r, model, tok = S.runner, S.model, S.tok
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


PROMPTS = ["The history of the Roman Empire", "def fibonacci(n):\n", "Question: A train travels 60 miles in 1.5 hours. What is its average speed?\nAnswer:",
           "The three most important things to know about photosynthesis are"]
CHAT_PROMPTS = ["Give me three tips for writing a clear email.", "Write a Python function that checks whether a string is a palindrome.",
                "A train travels 60 miles in 1.5 hours. What is its average speed? Explain briefly.", "What causes the seasons on Earth?"]


@torch.no_grad()
def nll(seqs):
    tot = n = 0.0
    for _, ids, _ in batches(seqs, 2):
        ids = ids.to(S.device)
        lp = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
        tot -= lp.gather(-1, ids[:, 1:, None]).sum().item()
        n += ids[:, 1:].numel()
    return tot / n


@torch.no_grad()
def samples():
    out = {}
    for p in CHAT_PROMPTS if args.chat else PROMPTS:
        text = p
        if args.chat:
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(S.device)
        g = model.generate(ids, max_new_tokens=160 if args.chat else 60, do_sample=False)
        out[p] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
    return out


@torch.no_grad()
def self_chat_seqs(n):
    """The model's own answers to UltraChat prompts through its real chat template, half with thinking on, packed into T-token rows."""
    import pandas as pd
    from common import _local_or_download
    from pathlib import Path
    path = _local_or_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset", allow_patterns=["data/test_sft-*.parquet"])
    df = pd.read_parquet(next(Path(path).rglob("test_sft-*.parquet")))
    prompts = [conv[0]["content"][:1500] for conv in df.messages[:n]]
    tok.padding_side = "left"
    stream = []
    for i in range(0, n, 16):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=(j % 2 == 0)) for j, p in enumerate(prompts[i:i + 16])]
        enc = tok(texts, return_tensors="pt", padding=True).to(S.device)
        g = model.generate(**enc, max_new_tokens=320, do_sample=False)
        for row, mask in zip(g, torch.cat([enc.attention_mask, torch.ones_like(g[:, enc.input_ids.shape[1]:])], 1)):
            stream += row[mask.bool()].tolist()
    T = S.T
    return [("selfchat", stream[k:k + T], [1] * T) for k in range(0, len(stream) - T, T)]


def sensitivity(seqs):
    """Per layer: Fisher matrix of the residual after that layer, and each token's sensitivity, from gradients of sampled log-probs."""
    L, d = r.L, cfg.hidden_size
    Fm = [torch.zeros(d, d, device=S.device) for _ in range(L)]
    tok_s = [[] for _ in range(L)]
    outs = {}
    def keep_out(i):
        def hook(mod, inp, out):
            out.retain_grad()
            outs[i] = out
        return hook
    hooks = [r.lm.layers[i].register_forward_hook(keep_out(i)) for i in range(L)]
    hooks.append(r.lm.embed_tokens.register_forward_hook(lambda mod, inp, out: out.requires_grad_(True)))
    n = 0
    for _, ids, _ in batches(seqs, 1):
        ids = ids.to(S.device)
        logits = model(ids).logits.float()
        with torch.no_grad():
            y = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]), 1).reshape(ids.shape)
        lp = F.log_softmax(logits, -1).gather(-1, y[..., None]).sum()
        lp.backward()
        for i in range(L):
            g = outs[i].grad[:, 1:].reshape(-1, d)
            Fm[i] += g.T @ g
            tok_s[i].append(g.pow(2).sum(1))
        n += g.shape[0]
        model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    return [x / n for x in Fm], [torch.cat(x) for x in tok_s]


prose = [x for x in S.data["eval"] if x[0] == "prose"]
calib = S.data["fit"][::max(1, len(S.data["fit"]) // args.calib_seqs)][:args.calib_seqs]
if args.self_chat:
    calib = calib + self_chat_seqs(args.self_chat)
    log(f"calibration: {len(calib)} rows including the model's own chat answers")
S.eval_batches()
result = {"model": args.model, "keep": args.keep, "before": {"prose_ppl": float(torch.tensor(nll(prose)).exp()), "samples": samples()}}
log(f"original prose ppl {result['before']['prose_ppl']:.2f}")
cfg = model.config
if args.shape > 0:
    with torch.enable_grad():
        fisher, tok_sens = sensitivity(calib)
    log("sensitivity measured")

m_old = cfg.intermediate_size
m = int(round(m_old * args.keep / 64) * 64)
for li in range(r.L):
    layer = r.lm.layers[li]
    mlp, norm = layer.mlp, layer.post_attention_layernorm
    # Inputs come from the model as pruned so far, so each layer corrects for the errors of the ones before it.
    Xs = []
    with torch.no_grad():
        for _, ids, _ in batches(calib, args.bs):
            emb, ctx = r.prepare(ids)
            _, rec = r.run(emb, 0, ctx, record={2 * li + 1}, positions=torch.tensor([0]))
            Xs.append(norm(rec[2 * li + 1][:, 1:]).reshape(-1, cfg.hidden_size))
        X = torch.cat(Xs)
        Y = mlp(X)
        Phi = F.silu(mlp.gate_proj(X)) * mlp.up_proj(X)
        Dn = mlp.down_proj.weight
        if args.shape > 0:
            # Error the model can't feel costs nothing: rank neurons by output-sensitive impact, refit with sensitive tokens weighted up.
            s = tok_sens[li] / tok_sens[li].mean()
            w = ((1 - args.shape) + args.shape * s).clamp(max=20)[:, None]
            imp = (w * Phi.pow(2)).mean(0) * (Dn * (fisher[li] @ Dn)).sum(0)
        else:
            w = torch.ones(len(X), 1, device=X.device)
            imp = Phi.pow(2).mean(0) * Dn.pow(2).sum(0)
        keep = imp.topk(m).indices.sort().values
        P = Phi[:, keep]
        A = P.T @ (w * P)
        A += args.ridge * A.diagonal().mean() * torch.eye(m, device=A.device)
        W = torch.linalg.solve(A, P.T @ (w * Y))
        r2 = 1 - (P @ W - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum()
        new = [nn.Linear(cfg.hidden_size, m, bias=False), nn.Linear(cfg.hidden_size, m, bias=False), nn.Linear(m, cfg.hidden_size, bias=False)]
        new[0].weight.data = mlp.gate_proj.weight[keep].clone()
        new[1].weight.data = mlp.up_proj.weight[keep].clone()
        new[2].weight.data = W.T.contiguous()
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = [x.to(S.device, mlp.gate_proj.weight.dtype) for x in new]
        mlp.intermediate_size = m
    log(f"L{li}: kept {m}/{m_old} neurons, layer R2 {r2.item():.3f}")
cfg.intermediate_size = m
if hasattr(cfg, "text_config"):
    cfg.text_config.intermediate_size = m

result["after"] = {"kl": S.kl(0), "prose_ppl": float(torch.tensor(nll(prose)).exp()), "samples": samples()}
n_params = sum(p.numel() for p in model.parameters())
result["params"] = n_params
log(f"pruned: KL {result['after']['kl']['all']['mean']:.3f}  prose ppl {result['after']['prose_ppl']:.2f}  params {n_params / 1e9:.2f}B")
model.to(torch.bfloat16).save_pretrained(args.out)
tok.save_pretrained(args.out)
save_json(result, f"{args.out}/build_report.json")
save_json(result, S.results / f"pruned_keep{int(args.keep * 100)}{args.tag}.json")
log("saved")
