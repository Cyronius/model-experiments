# How the stock 0.8B, our projected 0.8B and the 2B relate through their token embeddings (2026-10-19).
import glob
import json
import os

import gguf
import numpy as np
import torch
from safetensors import safe_open

HUB = os.path.expanduser("~/.cache/huggingface/hub")
KEY = "model.language_model.embed_tokens.weight"


def hf_embed(repo):
    f = glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/model.safetensors-00001-of-00001.safetensors")[0]
    with safe_open(f, "pt") as g:
        return g.get_tensor(KEY).float()


def gguf_embed(path):
    t = next(t for t in gguf.GGUFReader(path).tensors if t.name == "token_embd.weight")
    raw = np.ascontiguousarray(t.data).view(np.uint16).reshape(int(t.shape[1]), int(t.shape[0]))
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))


torch.set_num_threads(12)
E2, E08 = hf_embed("Qwen3.5-2B"), hf_embed("Qwen3.5-0.8B")
EH = gguf_embed(r"C:/code/model-experiments/models/manifold-compression/mc-proj-0.8b-tail.gguf")
V = E2.shape[0]
g = torch.Generator().manual_seed(0)
perm = torch.randperm(V, generator=g)
tr, te = perm[: int(0.9 * V)], perm[int(0.9 * V):]
out = {}


def fit(X, Y, lam=1e-4):
    """Ridge map X -> Y on train tokens; held-out R^2 (centered, pooled over output dims)."""
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X[tr] - mx, Y[tr] - my
    A = Xc.T.double() @ Xc.double()
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)
    W = torch.linalg.solve(A, Xc.T.double() @ Yc.double()).float()
    pred = (X[te] - mx) @ W + my
    r2 = 1 - (Y[te] - pred).pow(2).sum() / (Y[te] - Y[te].mean(0)).pow(2).sum()
    return W, r2.item()


def colmatch(A, B):
    """For each column of A: the best |correlation| with any single column of B."""
    Az = (A - A.mean(0)) / A.std(0)
    Bz = (B - B.mean(0)) / B.std(0)
    C = (Az.T @ Bz) / len(A)
    return C.abs().max(1).values


for name, Y in [("stock08", E08), ("ours08", EH)]:
    m = colmatch(Y, E2)
    out[f"{name}_best_single_2B_column_corr"] = {"median": m.median().item(), "frac_gt_0.9": (m > 0.9).float().mean().item(),
                                                  "frac_gt_0.5": (m > 0.5).float().mean().item()}

_, out["R2_2B_to_stock08"] = fit(E2, E08)
_, out["R2_stock08_to_2B"] = fit(E08, E2)
_, out["R2_2B_to_ours08"] = fit(E2, EH)
_, out["R2_ours08_to_stock08"] = fit(EH, E08)
_, out["R2_stock08_to_ours08"] = fit(E08, EH)

# Subspace of the 2B that each 0.8B embedding reads: column space of the 2048x1024 ridge map.
M08, _ = fit(E2, E08)
MH, _ = fit(E2, EH)
Q08, QH = torch.linalg.qr(M08)[0], torch.linalg.qr(MH)[0]
cos = torch.linalg.svdvals(Q08.T @ QH)
out["subspace_overlap_mean_cos2"] = cos.pow(2).mean().item()
out["subspace_overlap_random_expected"] = 1024 / 2048

# Energy of the 2B embeddings inside each subspace.
E2c = E2 - E2.mean(0)
tot = E2c.pow(2).sum()
for name, Q in [("stock08", Q08), ("ours08", QH)]:
    out[f"2B_embedding_energy_in_{name}_subspace"] = ((E2c @ Q).pow(2).sum() / tot).item()

print(json.dumps(out, indent=1))
