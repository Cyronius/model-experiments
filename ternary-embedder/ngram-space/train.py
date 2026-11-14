"""Plan B training (Stages 3 and 4): one arm per call.

Loss: mean over prefixes {256, 512, 1024, 4096} of 1 - cos (Plan A's H2 loss).
Table rows (u, theta, n-gram rows): SparseAdam. Dense parts: AdamW.
Regularization: lam * sum over rows touched in the batch of ||row - seed||^2 /
batch docs (u and n-gram rows); token dropout 10%. At most 10 epochs, early
stopping on validation loss (patience 1); the best epoch is kept.

Fixed hyper-parameters (set before any result): batch 512 docs; lr 1e-3 for
table rows; AdamW lr 3e-4 (V, mu, mixer, positions, scorer) and 1e-3 (MLP),
weight decay 0; constant learning rate; seed 0.

Usage (GPU, from ternary-embedder/):
  dibs run gpu -- <rocm python> -I ngram-space/train.py --arm S2_d1024 --d 1024 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from bmodel import Student, make_batch, prefix_cos_loss, inv_softplus  # noqa: E402

S1 = C.WORK / "stage1"
S2 = C.WORK / "stage2"
TIER = {"T1": 250_000, "T2": 1_000_000, "T3": 2_000_000}
BATCH = 512


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Data:
    """Token ids (uint16 memmap) and int8 targets for a set of Jaswanth rows."""

    def __init__(self, split, rows=None, device="cuda"):
        d = C.WORK / "stage0" / split
        self.ids = np.memmap(d / "ids.u16", dtype=np.uint16, mode="r")
        self.starts = np.load(d / "starts.npy")
        allrows = np.load(d / "rows.npy")
        if rows is None:
            rows = allrows
        self.rows = np.sort(rows)
        self.idx = np.searchsorted(allrows, self.rows)
        assert (allrows[self.idx] == self.rows).all()
        jw = C.WORK / "jaswanth"
        codes = np.load(jw / "codes.npy", mmap_mode="r")
        am = np.load(jw / "absmax.npy")
        self.codes = torch.from_numpy(np.ascontiguousarray(codes[self.rows])).to(device)
        self.am = torch.from_numpy(am[self.rows]).to(device)
        self.device = device

    def __len__(self):
        return len(self.rows)

    def seqs(self, sel):
        return [np.asarray(self.ids[self.starts[i]:self.starts[i + 1]]) for i in self.idx[sel]]

    def target(self, sel):
        s = torch.as_tensor(sel, device=self.device)
        y = self.codes[s].float() * (self.am[s, None] / 127.0)
        return torch.nn.functional.normalize(y, dim=-1)


def build(args, device):
    k2 = np.load(S1 / "ranked_n2_keys.npy")[:TIER[args.n2]] if args.n2 else None
    k3 = np.load(S1 / "ranked_n3_keys.npy")[:TIER[args.n3]] if args.n3 else None
    m = Student(d=args.d, n2_keys=k2, n3_keys=k3).to(device)
    return m, k2, k3


def c_rows(n, lo, hi):
    """Teacher coordinates c(g) for ranks [lo, hi): T1 file holds ranks 0..250k,
    T2 file ranks 250k..1M."""
    parts = []
    if lo < TIER["T1"]:
        parts.append(np.load(S2 / f"c_n{n}_T1.npy", mmap_mode="r")[lo:min(hi, TIER["T1"])])
    if hi > TIER["T1"]:
        a = max(lo, TIER["T1"]) - TIER["T1"]
        parts.append(np.load(S2 / f"c_n{n}_T2.npy", mmap_mode="r")[a:hi - TIER["T1"]])
    return np.asarray(np.concatenate(parts), np.float32)


def init_from_seeds(m, args, k2, k3, warm=None):
    """Initialize tables. Returns dict of seed tensors for the pull term."""
    from stage2 import seeds as SD   # noqa
    mu = np.load(S2 / "mu.npy")
    V = np.load(S2 / "V.npy")[:, :args.d]
    cv = np.load(S2 / "c_vocab.npy")[:, :args.d]
    w = np.load(S2 / "w_seed.npy")
    dev = m.V.device
    with torch.no_grad():
        if warm is not None:
            load_compatible(m, warm)
        else:
            m.V.copy_(torch.from_numpy(V))
            m.mu.copy_(torch.from_numpy(mu))
            m.theta.weight.copy_(inv_softplus(torch.from_numpy(w))[:, None])
            if args.init == "seed":
                m.u.weight.copy_(torch.from_numpy(cv))
            else:
                g = torch.Generator().manual_seed(0)
                std = float(cv[np.abs(cv).sum(1) > 0].std())
                m.u.weight.copy_(torch.randn(cv.shape, generator=g) * std)
        u_now = m.u.weight.detach().cpu().numpy()
        w_now = torch.nn.functional.softplus(m.theta.weight.detach()).squeeze(-1).cpu().numpy()
        # n-gram rows not in the warm start: seeded relative to the current tables
        for n, tab, keys in ((2, m.n2, k2), (3, m.n3, k3)):
            if tab is None:
                continue
            have = getattr(m, f"_warm_rows_n{n}", 0)
            if have >= len(keys):
                continue
            if args.ngram_init == "zero":
                continue
            new = keys[have:]
            c = c_rows(n, have, len(keys))[:, :args.d]
            if n == 2:
                r = SD.bigram_seeds(new, c, u_now, w_now)
            else:
                bk = m.n2.skeys.cpu().numpy()[np.argsort(m.n2.srow.cpu().numpy())] if m.n2 is not None else None
                br = m.n2.rows.weight.detach().cpu().numpy() if m.n2 is not None else None
                r = SD.trigram_seeds(new, c, u_now, w_now, bk, br)
            tab.rows.weight[have:] = torch.from_numpy(r * args.scale).to(dev)
    seeds = {"u": m.u.weight.detach().clone()}
    for n, tab in ((2, m.n2), (3, m.n3)):
        if tab is not None:
            seeds[f"n{n}"] = tab.rows.weight.detach().clone()
    return seeds


def load_compatible(m, path):
    """Warm start: copy every tensor present in the checkpoint; n-gram tables
    may grow (rows are in rank order, so a smaller tier is a prefix)."""
    sd = torch.load(path, map_location="cpu")
    own = m.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if k.endswith("rows.weight"):
            n = 2 if k.startswith("n2") else 3
            own[k][:len(v)] = v
            setattr(m, f"_warm_rows_n{n}", len(v))
        elif k.endswith("skeys") or k.endswith("srow"):
            continue
        elif own[k].shape == v.shape:
            own[k].copy_(v)
        elif k in ("u.weight", "V") and v.shape[-1] < own[k].shape[-1]:
            own[k][..., :v.shape[-1]] = v           # 512 -> 1024: nested coordinates
        else:
            raise ValueError(f"shape mismatch {k} {v.shape} {own[k].shape}")


def grow(m, args):
    """Apply ladder growth requested by args after a warm start."""
    if args.mlp and m.mlp is None:
        m.add_mlp()
    if args.chunk and not m.chunk:
        m.add_chunking(args.chunk)
    elif args.chunk and m.chunk != args.chunk:
        m.chunk = args.chunk
    while m.K < args.K:
        m.add_mixer_layer()


def param_groups(m):
    sparse = [m.u.weight, m.theta.weight] + [t.rows.weight for t in (m.n2, m.n3) if t is not None]
    ids = {id(p) for p in sparse}
    mlp = [p for p in (m.mlp.parameters() if m.mlp is not None else []) if id(p) not in ids]
    mlp_ids = {id(p) for p in mlp}
    dense = [p for p in m.parameters() if id(p) not in ids and id(p) not in mlp_ids]
    return sparse, dense, mlp


def pull(m, seeds, batch, lam, ndoc):
    if lam == 0:
        return 0.0
    tot = 0.0
    u_ids = torch.unique(batch["ids"])
    tot = ((m.u(u_ids) - seeds["u"][u_ids]) ** 2).sum()      # embedding call -> sparse grad
    for n, tab in ((2, m.n2), (3, m.n3)):
        if tab is None:
            continue
        keys = batch[f"n{n}k"]
        keys = keys[keys >= 0]
        row, found = tab.lookup(keys)
        r = torch.unique(row[found])
        tot = tot + ((tab.rows(r) - seeds[f"n{n}"][r]) ** 2).sum()
    return lam * tot / ndoc


@torch.no_grad()
def evaluate_loss(m, data, device, chunk_offset=0):
    m.eval()
    tot, n = 0.0, 0
    for b in range(0, len(data), 2048):
        sel = np.arange(b, min(b + 2048, len(data)))
        batch = make_batch(data.seqs(sel), device, offsets=np.full(len(sel), chunk_offset) if m.chunk else None)
        pred, _ = m(batch)
        tot += prefix_cos_loss(pred, data.target(sel), per_item=True).sum().item()
        n += len(sel)
    m.train()
    return tot / n


@torch.no_grad()
def encode(m, seqs, device, offset=0, bs=1024):
    m.eval()
    out = []
    for b in range(0, len(seqs), bs):
        part = seqs[b:b + bs]
        batch = make_batch(part, device, offsets=np.full(len(part), offset) if m.chunk else None)
        pred, _ = m(batch)
        out.append(torch.nn.functional.normalize(pred, dim=-1).cpu().numpy())
    m.train()
    return np.concatenate(out)


def train(args):
    sys.path.insert(0, "/home/cyrus/code/model-experiments/npu-kernel-coder/train")
    import memguard
    memguard.start(12)
    torch.manual_seed(0)
    device = args.device
    out = C.WORK / "runs" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    m, k2, k3 = build(args, device)
    warm = Path(args.warm) if args.warm else None
    if warm is not None:
        # build the warm model's structure first so its tensors load, then grow
        wargs = json.load(open(warm.parent / "args.json"))
        if wargs.get("mlp"):
            m.add_mlp()
        if wargs.get("chunk"):
            m.add_chunking(wargs["chunk"])
        for _ in range(wargs.get("K", 0)):
            m.add_mixer_layer()
    seeds = init_from_seeds(m, args, k2, k3, warm)
    grow(m, args)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    rows = np.load(C.WORK / "stage2" / "train300k_rows.npy") if args.rows == "300k" else \
        (np.load(C.WORK / "stage0" / "train_rows.npy") if args.rows == "all" else
         np.load(C.WORK / "stage3" / f"train{args.rows}_rows.npy"))
    if args.limit:
        rows = np.sort(rows)[:args.limit]
    tr = Data("train", rows, device)
    va = Data("val", None if not args.limit else np.load(C.WORK / "stage0" / "val_rows.npy")[:2000], device)
    log(f"arm {args.arm} train {len(tr)} val {len(va)} params "
        f"{sum(p.numel() for p in m.parameters()):,}")
    sparse, dense, mlp = param_groups(m)
    opt_s = torch.optim.SparseAdam(sparse, lr=args.lr_sparse)
    groups = [{"params": dense, "lr": args.lr_dense}]
    if mlp:
        groups.append({"params": mlp, "lr": args.lr_mlp})
    opt_d = torch.optim.AdamW(groups, weight_decay=0.0)
    hist = []
    v0 = evaluate_loss(m, va, device)
    hist.append({"epoch": 0, "val_loss": v0, "train_loss": None})
    log(f"epoch 0 val {v0:.5f}")
    best, best_ep = v0, 0
    if not (args.resume and (out / "train.json").exists()):
        torch.save(m.state_dict(), out / "best.pt")
    if args.epochs == 0:
        json.dump({"history": hist, "best_epoch": 0, "best_val_loss": v0}, open(out / "train.json", "w"), indent=1)
        return m
    rng = np.random.default_rng(0)
    start = 1
    if args.resume and (out / "last.pt").exists() and (out / "train.json").exists():
        # resume after an interruption: weights from last.pt, history from
        # train.json; the optimizer state starts fresh (not bit-identical)
        t = json.load(open(out / "train.json"))
        hist, best, best_ep = t["history"], t["best_val_loss"], t["best_epoch"]
        m.load_state_dict(torch.load(out / "last.pt", map_location=device))
        start = hist[-1]["epoch"] + 1
        rng = np.random.default_rng(1000 + start)
        log(f"resumed at epoch {start} (best epoch {best_ep} val {best:.5f})")
        if start - 1 - best_ep >= args.patience:
            start = args.epochs + 1
    for ep in range(start, args.epochs + 1):
        perm = rng.permutation(len(tr))
        tl, tn, t0 = 0.0, 0, time.time()
        for b in range(0, len(perm), BATCH):
            sel = np.sort(perm[b:b + BATCH])
            offs = rng.integers(0, m.chunk, len(sel)) if (m.chunk and args.random_offset) else \
                (np.zeros(len(sel), np.int64) if m.chunk else None)
            batch = make_batch(tr.seqs(sel), device, offsets=offs)
            drop = torch.rand(len(batch["ids"]), device=device) < args.dropout
            pred, _ = m(batch, drop=drop)
            loss = prefix_cos_loss(pred, tr.target(sel))
            tot = loss + pull(m, seeds, batch, args.lam, len(sel))
            opt_s.zero_grad(); opt_d.zero_grad()
            tot.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(dense + mlp, args.clip)
            opt_s.step(); opt_d.step()
            tl += loss.item() * len(sel); tn += len(sel)
        v = evaluate_loss(m, va, device)
        hist.append({"epoch": ep, "train_loss": tl / tn, "val_loss": v, "seconds": time.time() - t0})
        log(f"epoch {ep} train {tl/tn:.5f} val {v:.5f} ({time.time()-t0:.0f}s)")
        torch.save(m.state_dict(), out / "last.pt")
        if v < best:
            best, best_ep = v, ep
            torch.save(m.state_dict(), out / "best.pt")
        elif ep - best_ep >= args.patience:
            log("early stop")
            break
        json.dump({"history": hist, "best_epoch": best_ep, "best_val_loss": best}, open(out / "train.json", "w"), indent=1)
    # training loss of the best checkpoint on a 10k training sample, measured like validation
    m.load_state_dict(torch.load(out / "best.pt"))
    sub = Data("train", rows[np.random.default_rng(1).choice(len(rows), min(9204, len(rows)), replace=False)], device)
    tr_best = evaluate_loss(m, sub, device)
    json.dump({"history": hist, "best_epoch": best_ep, "best_val_loss": best,
               "best_train_loss_9204_sample_no_dropout": tr_best}, open(out / "train.json", "w"), indent=1)
    log(f"best epoch {best_ep} val {best:.5f} train(sample) {tr_best:.5f}")
    return m


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--n2", default="")
    ap.add_argument("--n3", default="")
    ap.add_argument("--init", default="seed", choices=["seed", "random"])
    ap.add_argument("--ngram_init", default="seed", choices=["seed", "zero"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--warm", default="")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--random_offset", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rows", default="300k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr_sparse", type=float, default=1e-3)
    ap.add_argument("--lr_dense", type=float, default=3e-4)
    ap.add_argument("--lr_mlp", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip on dense+mlp params; 0 = off")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
