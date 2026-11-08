# Gate 1 of the 9B-experts-on-a-small-trunk plan: how much of each 9B MLP can a 0.8B or 2B stream drive? (2026-11-05)
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "trunk-gate1-20261105")
WORK = os.path.join(OUT, "acts")
T, N_WIKI, N_CHAT, N_CODE = 512, 24, 24, 16
BIG, TRUNKS = "Qwen3.5-9B", ["Qwen3.5-2B", "Qwen3.5-0.8B"]
WINDOW = 3
torch.set_num_threads(12)
t0 = time.time()
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(f"[{time.time() - t0:.0f}s]", *a, flush=True)


def snap(repo):
    return glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/")[0]


def token_rows():
    tok = AutoTokenizer.from_pretrained(snap(BIG))
    for r in TRUNKS:
        assert open(snap(r) + "merges.txt", "rb").read() == open(snap(BIG) + "merges.txt", "rb").read(), r
        # The cached 2B snapshot has no tokenizer.json, so only snapshots that do are compared in full.
        if os.path.exists(snap(r) + "tokenizer.json"):
            assert tok.get_vocab() == AutoTokenizer.from_pretrained(snap(r)).get_vocab(), r
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
    return torch.tensor(seqs)


@torch.no_grad()
def capture(repo):
    ids = token_rows()
    big = repo == BIG
    m = AutoModelForCausalLM.from_pretrained(snap(repo), dtype=torch.bfloat16 if big else torch.float32).eval()
    layers = m.model.layers
    L, H, n = len(layers), m.config.get_text_config().hidden_size, len(ids) * (T - 1)
    xin = np.lib.format.open_memmap(os.path.join(WORK, f"{repo}_in.npy"), "w+", np.float16, (L, n, H))
    xout = np.lib.format.open_memmap(os.path.join(WORK, f"{repo}_out.npy"), "w+", np.float16, (L, n, H)) if big else None
    buf = {}

    def hook(i):
        def f(mod, args, out):
            buf[i] = (args[0][:, 1:].reshape(-1, H).float().numpy(), out[:, 1:].reshape(-1, H).float().numpy())
        return f

    hs = [layers[i].mlp.register_forward_hook(hook(i)) for i in range(L)]
    B = 4
    for b in range(0, len(ids), B):
        m(input_ids=ids[b:b + B])
        s = slice(b * (T - 1), (b + B) * (T - 1))
        for i in range(L):
            xin[i, s] = buf[i][0]
            if big:
                xout[i, s] = buf[i][1]
        log(repo, "rows", b + B, "of", len(ids))
    for h in hs:
        h.remove()
    xin.flush()
    if big:
        xout.flush()


def big_mlp(j):
    st = snap(BIG)
    idx = json.load(open(os.path.join(st, "model.safetensors.index.json")))["weight_map"]
    w = {}
    for part in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.language_model.layers.{j}.mlp.{part}.weight"
        with safe_open(os.path.join(st, idx[key]), "pt") as f:
            w[part] = f.get_tensor(key).float()
    return w


def r2_pooled(pred, true):
    d = true - true.mean(0)
    return (1 - (true - pred).pow(2).sum() / d.pow(2).sum()).item()


def r2_per_col(pred, true):
    d = true - true.mean(0)
    return 1 - (true - pred).pow(2).sum(0) / d.pow(2).sum(0).clamp(min=1e-12)


@torch.no_grad()
def fit(only=None):
    n_rows = N_WIKI + N_CHAT + N_CODE
    seq_of = torch.arange(n_rows).repeat_interleave(T - 1)
    te = seq_of % 4 == 3
    tr = ~te
    big_in = np.load(os.path.join(WORK, f"{BIG}_in.npy"), mmap_mode="r")
    big_out = np.load(os.path.join(WORK, f"{BIG}_out.npy"), mmap_mode="r")
    LB = big_in.shape[0]
    path = os.path.join(OUT, "gate1.json")
    report = json.load(open(path)) if only else {"tokens": int(len(seq_of)), "held_out_tokens": int(te.sum()), "trunks": {}}
    for trunk in TRUNKS:
        X_all = np.load(os.path.join(WORK, f"{trunk}_in.npy"), mmap_mode="r")
        LT = X_all.shape[0]
        solv = {}

        def solver(i):
            if i not in solv:
                X = torch.from_numpy(np.asarray(X_all[i])).float()
                mx = X[tr].mean(0)
                Xc = X[tr] - mx
                A = (Xc.T.double() @ Xc.double())
                A += 1e-3 * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)
                solv[i] = (X, mx, Xc, torch.linalg.cholesky(A))
            return solv[i]

        rows = report["trunks"][trunk]["layers"] if only else []
        for j in (only or range(LB)):
            Y = torch.from_numpy(np.asarray(big_in[j])).float()
            my = Y[tr].mean(0)
            Yc = Y[tr] - my
            centre = round(j * (LT - 1) / (LB - 1))
            best = None
            for i in range(max(0, centre - WINDOW), min(LT, centre + WINDOW + 1)):
                X, mx, Xc, Lc = solver(i)
                W = torch.cholesky_solve((Xc.T.double() @ Yc.double()), Lc).float()
                pred = (X[te] - mx) @ W + my
                r2 = r2_pooled(pred, Y[te])
                if best is None or r2 > best[1]:
                    best = (i, r2, pred)
            i, r2_in, Yhat = best
            w = big_mlp(j)
            Yt = Y[te]
            g_t, u_t = Yt @ w["gate_proj"].T, Yt @ w["up_proj"].T
            g_p, u_p = Yhat @ w["gate_proj"].T, Yhat @ w["up_proj"].T
            h_t, h_p = F.silu(g_t) * u_t, F.silu(g_p) * u_p
            out_true = torch.from_numpy(np.asarray(big_out[j])).float()[te]
            out_self, out_p = h_t @ w["down_proj"].T, h_p @ w["down_proj"].T
            act_w = h_t.abs().mean(0)
            r2_h = r2_per_col(h_p, h_t)
            row = {"big_layer": j, "trunk_layer": i, "R2_mlp_input": round(r2_in, 3),
                   "R2_gate_preact_median": round(r2_per_col(g_p, g_t).median().item(), 3),
                   "R2_up_preact_median": round(r2_per_col(u_p, u_t).median().item(), 3),
                   "R2_neuron_act_median": round(r2_h.median().item(), 3),
                   "R2_neuron_act_weighted": round((r2_h.clamp(min=-1) * act_w).sum().item() / act_w.sum().item(), 3),
                   "R2_mlp_output": round(r2_pooled(out_p, out_true), 3),
                   "cos_mlp_output_mean": round(F.cosine_similarity(out_p, out_true, dim=1).mean().item(), 3),
                   "check_R2_recomputed_vs_captured": round(r2_pooled(out_self, out_true), 4)}
            if only:
                rows[j] = row
            else:
                rows.append(row)
            log(trunk, row)
        med = float(np.median([r["R2_mlp_output"] for r in rows]))
        band = "pass" if med >= 0.6 else "dead" if med <= 0.3 else "mixed"
        report["trunks"][trunk] = {"median_R2_mlp_output": round(med, 3), "band": band, "layers": rows}
        log(trunk, "median R2 of MLP output", round(med, 3), band)
        json.dump(report, open(path, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2])
    else:
        fit([int(j) for j in sys.argv[2:]] or None)
