# Do single units (MLP neurons, attention heads) and their weights line up between the 2B and the stock 0.8B? (2026-10-19)
import glob
import json
import os
import time

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stock-relation-20261019")
# UNIT_SIZE: "dry" checks the whole script in about 2 minutes; "medium" fits beside other jobs (about 16 GB peak).
SIZE = os.environ.get("UNIT_SIZE", "full")
T, N_WIKI, N_CHAT, N_CODE = {"dry": (128, 4, 4, 4), "medium": (512, 8, 8, 8), "full": (512, 20, 16, 12)}[SIZE]
# Even layers only, so the recorded activations fit in RAM; all even layers use linear attention.
KEEP = list(range(0, 24, 2))
NULL_LAYERS, WHOLE_LAYERS = (2, 12, 22), (2, 8, 14, 20)
BS = 4
torch.set_num_threads(12)
torch.manual_seed(0)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:.0f}s] {msg}", flush=True)


def snap(repo):
    return glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/")[0]


tok = AutoTokenizer.from_pretrained(snap("Qwen3.5-2B"))
wiki = open(r"C:/code/model-experiments/bench/data/wikitext-2-raw/wiki.test.raw", encoding="utf-8").read()
ids = tok(wiki).input_ids
seqs = [ids[i * T:(i + 1) * T] for i in range(N_WIKI)]
chat = pd.read_parquet(glob.glob(f"{HUB}/datasets--HuggingFaceH4--ultrachat_200k/snapshots/*/data/test_sft-*.parquet")[0])
stream = []
for conv in chat.messages[:400]:
    stream += tok(tok.apply_chat_template(list(conv), tokenize=False, enable_thinking=False)).input_ids
    if len(stream) > N_CHAT * T:
        break
seqs += [stream[i * T:(i + 1) * T] for i in range(N_CHAT)]
mbpp = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{HUB}/datasets--google-research-datasets--mbpp/snapshots/*/full/*.parquet")])
code = tok("\n\n".join(f"# {t}\n{c}" for t, c in zip(mbpp.text, mbpp.code))).input_ids
seqs += [code[i * T:(i + 1) * T] for i in range(N_CODE)]
IDS = torch.tensor(seqs)
NS, PER = len(IDS), T - 1
seq_of = torch.arange(NS).repeat_interleave(PER)
TE = seq_of % 4 == 3
TR = ~TE


def flat(x):
    return x[:, 1:].reshape(-1, x.shape[-1]).half()


@torch.no_grad()
def record(repo):
    m = AutoModelForCausalLM.from_pretrained(snap(repo), dtype=torch.float32, attn_implementation="eager").eval()
    types = m.config.get_text_config().layer_types
    L = len(types)
    assert all(types[l] == "linear_attention" for l in KEEP)
    full = [l for l in range(L) if types[l] == "full_attention"]
    rec = {"h": {l: [] for l in KEEP}, "mid": {l: [] for l in KEEP}, "post": {l + 1: [] for l in KEEP},
           "lin": {l: [] for l in KEEP}, "att": {l: [] for l in full}}
    for l in KEEP:
        layer = m.model.layers[l]
        layer.mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: rec["h"][l].append(flat(a[0])))
        layer.post_attention_layernorm.register_forward_pre_hook(lambda mod, a, l=l: rec["mid"][l].append(flat(a[0])))
        layer.linear_attn.out_proj.register_forward_pre_hook(lambda mod, a, l=l: rec["lin"][l].append(flat(a[0])))
    for i in range(0, NS, BS):
        o = m(input_ids=IDS[i:i + BS], output_hidden_states=True, output_attentions=True)
        for j in rec["post"]:
            rec["post"][j].append(flat(o.hidden_states[j]))
        atts = [a for a in o.attentions if a is not None]
        assert len(atts) == len(full)
        for l, a in zip(full, atts):
            rec["att"][l].append(a[:, :, 1:, 1:].half())
        del o, atts
        log(f"{repo} batch {i // BS}")
    out = {k: {l: torch.cat(v) for l, v in rec[k].items()} for k in rec}
    w = {"gate": {l: m.model.layers[l].mlp.gate_proj.weight.detach().clone() for l in KEEP},
         "down": {l: m.model.layers[l].mlp.down_proj.weight.detach().clone() for l in KEEP},
         "pnorm": {l: m.model.layers[l].post_attention_layernorm.weight.detach().clone() for l in KEEP}}
    del m
    return out, w, types


A2, W2, types = record("Qwen3.5-2B")
A8, W8, types8 = record("Qwen3.5-0.8B")
assert types == types8


def zscore(X, mask):
    X = X[mask].float()
    return (X - X.mean(0)) / X.std(0).clamp(min=1e-6)


def ridge_map(X, Y, lam=1e-3):
    """Ridge map from X to Y fit on train rows, centered; returns W with Y ~ (X - mx) W + my."""
    X, Y = X[TR].float(), Y[TR].float()
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    A = Xc.T.double() @ Xc.double()
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)
    return torch.linalg.solve(A, Xc.T.double() @ Yc.double()).float()


def heldout_r2(X, Y, lam=1e-3):
    mx, my = X[TR].float().mean(0), Y[TR].float().mean(0)
    W = ridge_map(X, Y, lam)
    R = Y[TE].float() - ((X[TE].float() - mx) @ W + my)
    D = Y[TE].float() - Y[TE].float().mean(0)
    return (1 - R.pow(2).sum(0) / D.pow(2).sum(0).clamp(min=1e-9))


def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-9)


L = len(types)
res = {"tokens": int(TR.sum() + TE.sum()), "neurons": [], "full_heads": [], "linear_heads": []}

for l in KEEP:
    H2, H8 = A2["h"].pop(l), A8["h"].pop(l)
    Z2, Z8 = zscore(H2, TR), zscore(H8, TR)
    C = Z8.T @ Z2 / len(Z8)
    best = C.abs().argmax(1)
    sign = torch.sign(C.gather(1, best[:, None])[:, 0])
    Z2t, Z8t = zscore(H2, TE), zscore(H8, TE)
    held = (Z8t * Z2t[:, best]).mean(0) * sign
    row = {"layer": l, "median_best_corr": round(held.median().item(), 3),
           "frac_gt_0.5": round((held > 0.5).float().mean().item(), 3),
           "frac_gt_0.7": round((held > 0.7).float().mean().item(), 3),
           "frac_gt_0.9": round((held > 0.9).float().mean().item(), 3)}
    if l in NULL_LAYERS:
        Cn = Z8.T @ torch.roll(Z2, PER, 0) / len(Z8)
        row["null_median_best_corr_train"] = round(Cn.abs().max(1).values.median().item(), 3)
        row["real_median_best_corr_train"] = round(C.abs().max(1).values.median().item(), 3)
        del Cn
    if l in WHOLE_LAYERS:
        whole = heldout_r2(H2, H8)
        row["median_R2_from_whole_2B_layer"] = round(whole.median().item(), 3)
        row["median_best_single_corr2"] = round(held.clamp(min=0).pow(2).median().item(), 3)
    matched = held > 0.5
    S = best[matched].unique()
    imp = H2[TR].float().pow(2).mean(0) * W2["down"][l].pow(2).sum(0)
    top = torch.zeros(len(imp), dtype=torch.bool)
    top[imp.topk(W8["down"][l].shape[1]).indices] = True
    row["matched_pairs"] = int(matched.sum())
    row["distinct_2B_neurons_matched"] = int(len(S))
    row["matched_in_our_importance_top"] = round(top[S].float().mean().item(), 3) if len(S) else None
    row["chance_in_importance_top"] = round(top.float().mean().item(), 3)
    if matched.any():
        j = matched.nonzero()[:, 0]
        i = best[j]
        rnd = torch.randint(0, H2.shape[1], (len(j),))
        Wp = ridge_map(A2["post"][l + 1], A8["post"][l + 1])
        d8, s = W8["down"][l][:, j].T, sign[j]
        row["write_cos_matched"] = round((cos(W2["down"][l][:, i].T @ Wp, d8) * s).mean().item(), 3)
        row["write_abs_cos_random"] = round(cos(W2["down"][l][:, rnd].T @ Wp, d8).abs().mean().item(), 3)
        row["write_abs_cos_no_alignment"] = round(cos(W2["down"][l][:1024, i].T, d8).abs().mean().item(), 3)
        Wm = ridge_map(A2["mid"][l], A8["mid"][l])
        r8 = (1 + W8["pnorm"][l])[None] * W8["gate"][l][j]
        r2 = (1 + W2["pnorm"][l])[None] * W2["gate"][l][i]
        r2r = (1 + W2["pnorm"][l])[None] * W2["gate"][l][rnd]
        row["read_cos_matched"] = round(cos(r8 @ Wm.T, r2).mean().item(), 3)
        row["read_abs_cos_random"] = round(cos(r8 @ Wm.T, r2r).abs().mean().item(), 3)
    res["neurons"].append(row)
    log(f"neurons {row}")

for l, M8 in A8["att"].items():
    M2 = A2["att"][l]
    n, H, q, _ = M8.shape
    tri = torch.tril(torch.ones(q, q, dtype=torch.bool))

    def heads(M):
        X = M.float()[:, :, tri].permute(1, 0, 2).reshape(H, -1)
        return (X - X.mean(1, keepdim=True)) / X.std(1, keepdim=True)

    X8, X2, X2n = heads(M8), heads(M2), heads(torch.roll(M2, 1, 0))
    S = X8 @ X2.T / X8.shape[1]
    Sn = X8 @ X2n.T / X8.shape[1]
    r, c = linear_sum_assignment(-S.numpy())
    row = {"layer": l, "identity_mean_corr": round(S.diagonal().mean().item(), 3),
           "best_assignment_mean_corr": round(S[r, c].mean().item(), 3),
           "best_assignment": c.tolist(), "null_shifted_mean_corr": round(Sn[r, c].mean().item(), 3),
           "matrix": [[round(v, 2) for v in s] for s in S.tolist()]}
    res["full_heads"].append(row)
    log(f"full heads layer {l}: identity {row['identity_mean_corr']} best {row['best_assignment_mean_corr']} "
        f"null {row['null_shifted_mean_corr']} assignment {row['best_assignment']}")

for l, Y in A8["lin"].items():
    X = A2["lin"][l]
    nh, hd = 16, X.shape[1] // 16
    R = torch.zeros(nh, nh)
    for b in range(nh):
        r2 = heldout_r2(X[:, b * hd:(b + 1) * hd], Y)
        for a in range(nh):
            R[a, b] = 1 - (1 - r2[a * hd:(a + 1) * hd]).mean()
    whole = heldout_r2(X, Y)
    whole_h = torch.stack([whole[a * hd:(a + 1) * hd].mean() for a in range(nh)])
    r, c = linear_sum_assignment(-R.numpy())
    row = {"layer": l, "identity_mean_R2": round(R.diagonal().mean().item(), 3),
           "best_assignment_mean_R2": round(R[r, c].mean().item(), 3), "best_assignment": c.tolist(),
           "mean_R2_from_all_2B_heads": round(whole_h.mean().item(), 3)}
    res["linear_heads"].append(row)
    log(f"linear heads {row}")

json.dump(res, open(os.path.join(OUT, f"unit_relation_{SIZE}.json"), "w"), indent=1)
log("done")
