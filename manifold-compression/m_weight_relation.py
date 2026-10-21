# Direct weight comparison of the 2B and the stock 0.8B: parameters whose shape does not depend on model width (2026-10-19).
import glob
import json
import os

import torch
from safetensors import safe_open

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stock-relation-20261019")
PRE = "model.language_model.layers"
LINEAR = [l for l in range(24) if l % 4 != 3]
FULL = [l for l in range(24) if l % 4 == 3]
torch.manual_seed(0)


def load(repo):
    f = glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/model.safetensors-00001-of-00001.safetensors")[0]
    g = safe_open(f, "pt")
    return lambda k: g.get_tensor(k).float()


W2, W8 = load("Qwen3.5-2B"), load("Qwen3.5-0.8B")


def corr(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def tap_centered(w):
    w = w.reshape(w.shape[0], -1)
    return w - w.mean(0)


# Each feature maps a layer's weights to a vector whose entries are indexed by heads or channels, not by model width.
FEATURES = {
    "linear": {
        "A_log": lambda W, l: W(f"{PRE}.{l}.linear_attn.A_log"),
        "dt_bias": lambda W, l: W(f"{PRE}.{l}.linear_attn.dt_bias"),
        "conv1d_q": lambda W, l: tap_centered(W(f"{PRE}.{l}.linear_attn.conv1d.weight")[:2048]),
        "conv1d_k": lambda W, l: tap_centered(W(f"{PRE}.{l}.linear_attn.conv1d.weight")[2048:4096]),
        "conv1d_v": lambda W, l: tap_centered(W(f"{PRE}.{l}.linear_attn.conv1d.weight")[4096:]),
        "gated_norm": lambda W, l: W(f"{PRE}.{l}.linear_attn.norm.weight"),
        "in_qkv_row_norms": lambda W, l: W(f"{PRE}.{l}.linear_attn.in_proj_qkv.weight").norm(dim=1).clamp(min=1e-8).log(),
        "in_z_row_norms": lambda W, l: W(f"{PRE}.{l}.linear_attn.in_proj_z.weight").norm(dim=1).clamp(min=1e-8).log(),
        "out_col_norms": lambda W, l: W(f"{PRE}.{l}.linear_attn.out_proj.weight").norm(dim=0).clamp(min=1e-8).log(),
    },
    "full": {
        "q_norm": lambda W, l: W(f"{PRE}.{l}.self_attn.q_norm.weight"),
        "k_norm": lambda W, l: W(f"{PRE}.{l}.self_attn.k_norm.weight"),
        "q_row_norms": lambda W, l: W(f"{PRE}.{l}.self_attn.q_proj.weight").norm(dim=1).clamp(min=1e-8).log(),
        "k_row_norms": lambda W, l: W(f"{PRE}.{l}.self_attn.k_proj.weight").norm(dim=1).clamp(min=1e-8).log(),
        "v_row_norms": lambda W, l: W(f"{PRE}.{l}.self_attn.v_proj.weight").norm(dim=1).clamp(min=1e-8).log(),
        "o_col_norms": lambda W, l: W(f"{PRE}.{l}.self_attn.o_proj.weight").norm(dim=0).clamp(min=1e-8).log(),
    },
}

res = {}
for kind, layers in (("linear", LINEAR), ("full", FULL)):
    for name, fn in FEATURES[kind].items():
        f2 = {l: fn(W2, l) for l in layers}
        f8 = {l: fn(W8, l) for l in layers}
        same = [corr(f8[l], f2[l]) for l in layers]
        other = [corr(f8[l], f2[m]) for l in layers for m in layers if m != l]
        perm = []
        for l in layers:
            x = f8[l]
            perm.append(corr(x[torch.randperm(len(x))], f2[l]))
        res[name] = {"same_layer_mean": round(sum(same) / len(same), 3), "same_layer_min": round(min(same), 3),
                     "other_layer_mean": round(sum(other) / len(other), 3),
                     "shuffled_mean": round(sum(perm) / len(perm), 3), "per_layer": [round(v, 3) for v in same]}
        print(f"{name:18s} same layer {res[name]['same_layer_mean']:+.3f} (min {res[name]['same_layer_min']:+.3f})  "
              f"other layer {res[name]['other_layer_mean']:+.3f}  shuffled {res[name]['shuffled_mean']:+.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, "weight_relation.json"), "w"), indent=1)
