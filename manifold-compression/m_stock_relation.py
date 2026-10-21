# Is the stock 0.8B's residual stream a linear readout of the 2B's, and does our PCA basis keep that readout? (2026-10-19)
import glob
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stock-relation-20261019")
T, N_WIKI, N_CHAT, N_CODE = 512, 24, 24, 16
torch.set_num_threads(12)
t0 = time.time()


def snap(repo):
    return glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/")[0]


tok = AutoTokenizer.from_pretrained(snap("Qwen3.5-2B"))
assert tok.get_vocab() == AutoTokenizer.from_pretrained(snap("Qwen3.5-0.8B")).get_vocab()

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
dom = ["wiki"] * N_WIKI + ["chat"] * N_CHAT + ["code"] * N_CODE
print(f"[{time.time() - t0:.0f}s] {IDS.shape} tokens", flush=True)


@torch.no_grad()
def states(repo):
    m = AutoModelForCausalLM.from_pretrained(snap(repo), dtype=torch.float32).eval()
    hs = []
    for i in range(0, len(IDS), 8):
        o = m(input_ids=IDS[i:i + 8], output_hidden_states=True)
        hs.append(torch.stack([h[:, 1:].reshape(-1, h.shape[-1]).half() for h in o.hidden_states]))
        print(f"[{time.time() - t0:.0f}s] {repo} batch {i // 8}", flush=True)
    E = m.get_input_embeddings().weight.detach().clone()
    del m
    return torch.cat(hs, 1), E


H2, E2 = states("Qwen3.5-2B")
H8, _ = states("Qwen3.5-0.8B")

import gguf  # noqa: E402
import numpy as np  # noqa: E402

t = next(t for t in gguf.GGUFReader(r"C:/code/model-experiments/models/manifold-compression/mc-proj-0.8b-tail.gguf").tensors
         if t.name == "token_embd.weight")
raw = np.ascontiguousarray(t.data).view(np.uint16).reshape(int(t.shape[1]), int(t.shape[0]))
EH = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))
# Our embeddings are exactly E2 @ P, so least squares recovers the shared basis P.
P = torch.linalg.solve(E2.T.double() @ E2.double(), E2.T.double() @ EH.double()).float()
orth = (P.T @ P - torch.eye(P.shape[1])).abs().max().item()

per_seq = T - 1
seq_of = torch.arange(len(IDS)).repeat_interleave(per_seq)
te_mask = (seq_of % 4 == 3)
tr_mask = ~te_mask


def ridge_r2(X, Y, lam=1e-3):
    X, Y = X.float(), Y.float()
    mx, my = X[tr_mask].mean(0), Y[tr_mask].mean(0)
    Xc, Yc = X[tr_mask] - mx, Y[tr_mask] - my
    A = Xc.T.double() @ Xc.double()
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)
    W = torch.linalg.solve(A, Xc.T.double() @ Yc.double()).float()
    R = Y[te_mask] - ((X[te_mask] - mx) @ W + my)
    D = Y[te_mask] - Y[te_mask].mean(0)
    pooled = 1 - R.pow(2).sum() / D.pow(2).sum()
    per_dim = (1 - R.pow(2).sum(0) / D.pow(2).sum(0)).clamp(min=-1).mean()
    return W, pooled.item(), per_dim.item()


Q_P = torch.linalg.qr(P)[0]
rows = []
for l in range(H2.shape[0]):
    X, Y = H2[l], H8[l]
    W, r2, r2d = ridge_r2(X, Y)
    _, r2p, r2pd = ridge_r2(X.float() @ P, Y)
    _, r2rev, _ = ridge_r2(Y, X)
    Qw = torch.linalg.qr(W)[0]
    Xf = X.float()
    rows.append({"layer": l, "R2_2B_to_stock08": round(r2, 3), "R2_2B_to_stock08_per_dim": round(r2d, 3),
                 "R2_ourbasis_to_stock08": round(r2p, 3), "R2_ourbasis_to_stock08_per_dim": round(r2pd, 3),
                 "R2_stock08_to_2B": round(r2rev, 3),
                 "energy_kept_by_our_basis": round(((Xf @ P).pow(2).sum() / Xf.pow(2).sum()).item(), 3),
                 "overlap_readout_vs_our_basis_cos2": round(torch.linalg.svdvals(Qw.T @ Q_P).pow(2).mean().item(), 3)})
    print(rows[-1], flush=True)

# Layer correspondence: which 2B layer best predicts each stock 0.8B layer.
best = []
for j in range(0, H8.shape[0], 2):
    scores = {i: ridge_r2(H2[i], H8[j])[1] for i in range(max(0, j - 6), min(H2.shape[0], j + 7))}
    i_best = max(scores, key=scores.get)
    best.append({"stock08_layer": j, "best_2B_layer": i_best, "R2_best": round(scores[i_best], 3),
                 "R2_same_layer": round(scores[j], 3)})
    print(best[-1], flush=True)

json.dump({"tokens_per_model": int(H2.shape[1]), "P_orthonormal_maxerr": orth, "layers": rows, "layer_match": best,
           "note": "hidden_states[-1] is after the final norm; position 0 of every row dropped; held-out = every 4th sequence"},
          open(os.path.join(OUT, "stream_relation.json"), "w"), indent=1)
print(f"[{time.time() - t0:.0f}s] done", flush=True)
