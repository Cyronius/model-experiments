"""FLUX enricher v2 on curated DiffusionDB append-form shards: data, train, score (2026-11-14)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

CTX, WIN, TAIL, WIDTH = 64, 32, 9, 3072
LEN_MIN = 12  # shortest changed-row count in the data: 3 detail rows + 9 template rows
N_LEN = WIN - LEN_MIN + 1


def split_of(group_id: str) -> str:
    h = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else "validation" if h < 95 else "test"


def data(args) -> None:
    tags = {}
    for spec in args.tag:
        name, path = spec.split("=", 1)
        tags.update({json.loads(line)["id"]: name for line in open(path, encoding="utf-8")})
    shards = sorted(args.shards.glob("shard-*.json"))[: args.limit_shards or None]
    groups, items, shard_rows, dropped = [], [], [], {"context": 0, "window": 0, "single": 0}
    row0 = 0
    for s, path in enumerate(shards):
        meta = json.loads(path.read_text(encoding="utf-8"))
        n_rows = 0
        for g in meta["groups"]:
            n_rows = max(n_rows, g["base_offset"] + g["base_rows"], *[d["offset"] + d["rows"] for d in g["details"]])
            if not g["details"]:
                continue
            k = g["details"][0]["first_changed_row"]
            if k - 3 > CTX:
                dropped["context"] += len(g["details"])
                continue
            entry = {"id": g["id"], "base": g["base"], "split": split_of(g["id"]), "tag": tags.get(g["id"], "9b"),
                     "shard": s, "k": k, "ctx_start": row0 + g["base_offset"] + 3, "ctx_len": k - 3,
                     "items": [], "details": []}
            for d in g["details"]:
                assert d["first_changed_row"] == k
                if d["rows"] > WIN:
                    dropped["window"] += 1
                    continue
                entry["items"].append(len(items))
                entry["details"].append(d["text"])
                items.append([len(groups), entry["ctx_start"], k - 3, row0 + d["offset"], d["rows"]])
            if len(entry["items"]) < 2:
                dropped["single"] += len(entry["items"])
                for _ in entry["items"]:
                    items.pop()
                continue
            groups.append(entry)
        shard_rows.append(n_rows)
        row0 += n_rows
    items_t = torch.tensor(items, dtype=torch.int64)
    print(f"{len(shards)} shards, {row0} rows, {len(groups)} groups, {len(items)} items, dropped {dropped}", flush=True)

    def train_mask(s: int, start: int, n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        for g in groups:
            if g["shard"] == s and g["split"] == "train":
                m[g["ctx_start"] - start: g["ctx_start"] - start + g["ctx_len"]] = True
                for i in g["items"]:
                    m[items[i][3] - start: items[i][3] - start + items[i][4]] = True
        return m

    # One pass: shifted sums give a stable mean and covariance of train rows.
    shift, total, s1, s2, start = None, 0, torch.zeros(WIDTH, dtype=torch.float64), None, 0
    s2 = torch.zeros(WIDTH, WIDTH, dtype=torch.float64)
    t0 = time.time()
    for s, path in enumerate(shards):
        rows = load_file(str(path.with_suffix(".safetensors")))["rows"].float()
        x = rows[torch.from_numpy(train_mask(s, start, len(rows)))]
        if shift is None:
            shift = x.mean(0)
        x = x - shift
        total += len(x)
        s1 += x.sum(0).double()
        s2 += (x.T @ x).double()
        start += shard_rows[s]
        if s % 10 == 0:
            print(f"stats shard {s}, {total} rows, {time.time() - t0:.0f} s", flush=True)
    m1 = s1 / total
    cov = s2 / total - torch.outer(m1, m1)
    mean = (shift.double() + m1).float()
    std = cov.diagonal().clamp_min(1e-6).sqrt().float()
    corr = cov / torch.outer(std.double(), std.double())
    eig, vec = torch.linalg.eigh(corr)
    eig, vec = eig.flip(0).clamp_min(0), vec.flip(1)
    frac = (eig.cumsum(0) / eig.sum()).tolist()
    table = {n: frac[n - 1] for n in (64, 128, 256, 384, 512, 768, 1024)}
    print("explained variance: " + ", ".join(f"{n}={v:.4f}" for n, v in table.items()), flush=True)
    p = args.pca or next((n for n in (256, 384, 512) if table[n] >= 0.95), 512)
    basis = vec[:, :p].T.float().contiguous()
    scale = eig[:p].sqrt().float()

    args.output.mkdir(parents=True, exist_ok=True)
    start = 0
    for s, path in enumerate(shards):
        rows = load_file(str(path.with_suffix(".safetensors")))["rows"].float()
        z = ((rows - mean) / std) @ basis.T
        save_file({"z": z.to(torch.float16).contiguous()}, str(args.output / f"z-{s:05d}.safetensors"))
        start += shard_rows[s]
    save_file({"mean": mean, "std": std, "basis": basis, "scale": scale, "items": items_t},
              str(args.output / "stats.safetensors"))
    counts = {sp: {"groups": sum(g["split"] == sp for g in groups),
                   "items": sum(len(g["items"]) for g in groups if g["split"] == sp),
                   "tags": {t: sum(g["split"] == sp and g["tag"] == t for g in groups) for t in {"9b", *tags.values()}}}
              for sp in ("train", "validation", "test")}
    info = {"shards": [str(p) for p in shards], "shard_rows": shard_rows, "rows": row0, "pca": p,
            "explained_variance": table, "dropped": dropped, "counts": counts, "train_rows_for_stats": total,
            "limits": {"context": CTX, "window": WIN, "len_min": LEN_MIN}}
    (args.output / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (args.output / "groups.json").write_text(json.dumps(groups, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(counts), f"pca={p}", flush=True)


class Data:
    def __init__(self, path: Path, device: str):
        info = json.loads((path / "info.json").read_text(encoding="utf-8"))
        st = load_file(str(path / "stats.safetensors"))
        self.mean, self.std, self.basis = st["mean"], st["std"], st["basis"]
        self.scale = st["scale"].to(device)
        self.items = st["items"].to(device)
        self.groups = json.loads((path / "groups.json").read_text(encoding="utf-8"))
        self.z = torch.cat([load_file(str(path / f"z-{s:05d}.safetensors"))["z"]
                            for s in range(len(info["shards"]))]).to(device)
        self.p, self.device, self.info = info["pca"], device, info
        split = torch.tensor([("train", "validation", "test").index(g["split"]) for g in self.groups], device=device)
        self.item_split = split[self.items[:, 0]]

    def gather(self, idx: torch.Tensor, whiten: bool = True):
        it = self.items[idx]
        ar_c, ar_w = torch.arange(CTX, device=self.device), torch.arange(WIN, device=self.device)
        cm, wm = ar_c < it[:, 2, None], ar_w < it[:, 4, None]
        ctx = self.z[torch.where(cm, it[:, 1, None] + ar_c, 0)].float() * cm[..., None]
        win = self.z[torch.where(wm, it[:, 3, None] + ar_w, 0)].float() * wm[..., None]
        if whiten:
            ctx, win = ctx / self.scale, win / self.scale
        return ctx, cm, win, wm, it[:, 4]

    def to_raw(self, z: torch.Tensor) -> torch.Tensor:
        return (z.float().cpu() @ self.basis) * self.std + self.mean


def time_embed(t: torch.Tensor) -> torch.Tensor:
    freqs = torch.exp(torch.arange(32, device=t.device) * (-math.log(1000.0) / 32))
    return torch.cat([torch.sin(t[:, None] * 1000 * freqs), torch.cos(t[:, None] * 1000 * freqs)], -1)


def encoder(dim: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Enricher(nn.Module):
    """Clean base rows as context, masked noisy window rows to denoise, and a length head on the context."""

    def __init__(self, io: int, dim: int = 512, layers: int = 8, heads: int = 8):
        super().__init__()
        self.inp, self.ctx, self.out = nn.Linear(io, dim), nn.Linear(io, dim), nn.Linear(dim, io)
        self.pos = nn.Parameter(torch.randn(WIN, dim) * 0.02)
        # The 9 template rows sit at the window end, so rows also get a position counted from the end.
        self.rpos = nn.Parameter(torch.randn(WIN, dim) * 0.02)
        self.ctx_pos = nn.Parameter(torch.randn(CTX, dim) * 0.02)
        self.kind = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(64, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.body, self.norm = encoder(dim, heads, layers), nn.LayerNorm(dim)
        self.len_body, self.len_norm = encoder(dim, heads, 2), nn.LayerNorm(dim)
        self.len_out = nn.Linear(dim, N_LEN)

    def embed_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.ctx(ctx) + self.ctx_pos[: ctx.shape[1]] + self.kind[1]

    def forward(self, x, t, ctx, cm, wm):
        rev = (wm.sum(1, keepdim=True) - 1 - torch.arange(WIN, device=x.device)).clamp_min(0)
        h = self.inp(x) + self.pos + self.rpos[rev] + self.kind[0] + self.time(time_embed(t))[:, None]
        seq = torch.cat([self.embed_ctx(ctx), h], 1)
        out = self.body(seq, src_key_padding_mask=~torch.cat([cm, wm], 1))[:, ctx.shape[1]:]
        return self.out(self.norm(out))

    def length_logits(self, ctx, cm):
        cls = self.kind[2].expand(len(ctx), 1, -1)
        seq = torch.cat([cls, self.embed_ctx(ctx)], 1)
        pad = ~torch.cat([torch.ones_like(cm[:, :1]), cm], 1)
        return self.len_out(self.len_norm(self.len_body(seq, src_key_padding_mask=pad)[:, 0]))


def masked_mse(pred, target, wm):
    return ((pred - target).square().mean(-1) * wm).sum() / wm.sum()


def train(args) -> None:
    torch.manual_seed(args.seed)
    d = Data(args.data, args.device)
    tr = torch.nonzero(d.item_split == 0).flatten()
    va = torch.nonzero(d.item_split == 1).flatten()
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    va_batch = va[torch.randperm(len(va), generator=gen, device=args.device)[:1024]]
    model = Enricher(d.p, args.dim, args.layers, args.heads).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(tr)} train items, {len(va)} validation items, {n_params / 1e6:.1f}M parameters, mode {args.mode}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)
    amp = torch.autocast("cuda", torch.bfloat16) if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    start_step, log = 0, []
    if args.resume and args.output.exists():
        ck = torch.load(args.output, weights_only=False, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step, log = ck["step"], ck["log"]
        print(f"resumed at step {start_step}", flush=True)

    def losses(idx, g):
        ctx, cm, x1, wm, n = d.gather(idx)
        with amp:
            if args.mode == "flow":
                t = torch.rand(len(idx), device=args.device, generator=g)
                noise = torch.randn(x1.shape, device=args.device, generator=g)
                xt = (t[:, None, None] * noise + (1 - t[:, None, None]) * x1) * wm[..., None]
                # Classifier-free guidance: some steps train without the base rows.
                keep = torch.rand(len(idx), device=args.device, generator=g) >= args.ctx_drop
                pred = model(xt, t, ctx * keep[:, None, None], cm & keep[:, None], wm)
            else:
                pred = model(torch.zeros_like(x1), torch.zeros(len(idx), device=args.device), ctx, cm, wm)
            rows = masked_mse(pred.float(), x1, wm)
            length = F.cross_entropy(model.length_logits(ctx, cm).float(), (n - LEN_MIN).clamp(0, N_LEN - 1))
        return rows, length

    def save(step):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "log": log, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "p": d.p}, tmp)
        tmp.replace(args.output)

    t0, deadline = time.time(), time.time() + args.time_limit * 60 if args.time_limit else math.inf
    for step in range(start_step, args.steps):
        model.train()
        rows, length = losses(tr[torch.randint(len(tr), (args.batch,), device=args.device, generator=gen)], gen)
        loss = rows + args.length_weight * length
        rows, length = rows.detach(), length.detach()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = step == args.steps - 1 or time.time() > deadline
        if step % args.every == 0 or last:
            model.eval()
            with torch.no_grad():
                vr, vl = losses(va_batch, torch.Generator(device=args.device).manual_seed(0))
            rate = (step + 1 - start_step) / (time.time() - t0)
            log.append({"step": step, "rows": float(rows), "length": float(length), "val_rows": float(vr),
                        "val_length": float(vl), "steps_per_s": rate})
            print(f"step {step} rows {float(rows):.4f} len {float(length):.3f} val_rows {float(vr):.4f} "
                  f"val_len {float(vl):.3f} {rate:.2f} steps/s", flush=True)
        if (step + 1) % args.save_every == 0 or last:
            save(step + 1)
        if last:
            break


def load_model(path: Path, device: str):
    ck = torch.load(path, weights_only=False, map_location=device)
    a = ck["args"]
    model = Enricher(ck["p"], a["dim"], a["layers"], a["heads"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), a["mode"], ck["step"]


@torch.no_grad()
def generate(model, mode, ctx, cm, n, steps, gen, guidance: float = 1.0):
    wm = torch.arange(WIN, device=ctx.device) < n[:, None]
    x = torch.randn(len(ctx), WIN, ctx.shape[-1], device=ctx.device, generator=gen) * wm[..., None]
    if mode != "flow":
        return model(torch.zeros_like(x), torch.zeros(len(x), device=x.device), ctx, cm, wm) * wm[..., None]
    for i in range(steps):
        t = torch.full((len(x),), 1 - i / steps, device=x.device)
        out = model(x, t, ctx, cm, wm)
        if guidance != 1.0:
            free = model(x, t, torch.zeros_like(ctx), torch.zeros_like(cm), wm)
            out = free + guidance * (out - free)
        # x0 prediction: velocity = (x - x0) / t; the last step lands on the x0 estimate.
        x = (x - (x - out) / t[:, None, None] / steps) * wm[..., None]
    return x


def descriptor(win: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Mean of the detail rows plus the 9 template-tail rows: a fixed-size view of a variable-length window."""
    ar = torch.arange(WIN, device=win.device)
    content = (ar < (n - TAIL)[:, None]).float()
    mean = (win * content[..., None]).sum(1) / content.sum(1, keepdim=True)
    tail = win.gather(1, ((n - TAIL)[:, None] + torch.arange(TAIL, device=win.device))[..., None].expand(-1, -1, win.shape[-1]))
    return torch.cat([mean[:, None], tail], 1).flatten(1)


@torch.no_grad()
def score(args) -> None:
    d = Data(args.data, args.device)
    model, mode, step = load_model(args.checkpoint, args.device)
    reg = load_model(args.regression, args.device)[0] if args.regression else None
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr = torch.nonzero(d.item_split == 0).flatten()
    # Mean baseline: position-wise train mean window for each length, in normalized PCA space.
    mean_win = torch.zeros(N_LEN, WIN, d.p, device=args.device)
    n_tr = d.items[tr, 4]
    for li in range(N_LEN):
        idx = tr[n_tr == li + LEN_MIN]
        for c in range(0, len(idx), 8192):
            mean_win[li] += d.gather(idx[c:c + 8192], whiten=False)[2].sum(0)
        mean_win[li] /= max(len(idx), 1)
    groups = [i for i, g in enumerate(d.groups) if g["split"] == args.split]
    if args.max_groups:
        groups = sorted(rng.choice(groups, min(args.max_groups, len(groups)), replace=False).tolist())
    teach = {}
    for gi in groups:
        idx = torch.tensor(d.groups[gi]["items"], device=args.device)
        _, _, w, _, n = d.gather(idx, whiten=False)
        teach[gi] = descriptor(w, n)
    rows, length_err = [], []
    image_groups = set(groups[:: max(1, len(groups) // args.image_groups)][: args.image_groups]) if args.image_groups else set()
    image_out, image_index = {}, []
    S = args.samples
    for c in range(0, len(groups), args.chunk):
        chunk = groups[c:c + args.chunk]
        first = torch.tensor([d.groups[gi]["items"][0] for gi in chunk], device=args.device)
        ctx, cm, _, _, _ = d.gather(first)
        logits = model.length_logits(ctx, cm)
        for j, gi in enumerate(chunk):
            true_n = d.items[torch.tensor(d.groups[gi]["items"], device=args.device), 4]
            length_err.append(float((logits[j].argmax() + LEN_MIN - true_n).abs().float().mean()))
        n = torch.multinomial(logits.softmax(-1), S, replacement=True, generator=gen).flatten() + LEN_MIN
        ctx_r, cm_r = ctx.repeat_interleave(S, 0), cm.repeat_interleave(S, 0)
        out = {f"enricher_g{w:g}": generate(model, mode, ctx_r, cm_r, n, args.steps, gen, w) * d.scale
               for w in args.guidance}
        out["mean"] = mean_win[n - LEN_MIN]
        if reg is not None:
            out["regression"] = generate(reg, "regress", ctx_r, cm_r, n, args.steps, gen) * d.scale
        descs = {k: descriptor(v, n) for k, v in out.items()}
        for j, gi in enumerate(chunk):
            own = teach[gi]
            dist_tt = torch.cdist(own, own)
            off = ~torch.eye(len(own), dtype=torch.bool, device=own.device)
            D = dist_tt[off].median()
            others = rng.choice([g for g in groups if g != gi], min(args.others, len(groups) - 1), replace=False)
            pool = torch.cat([teach[o] for o in others])
            row = {"group": d.groups[gi]["id"], "tag": d.groups[gi]["tag"], "D": float(D), "teachers": len(own)}
            # Reference: each real enrichment scored against the other enrichments of its group (a perfect sampler).
            nn_t = dist_tt.masked_fill(~off, float("inf")).min(1).values
            row["held_out_teacher"] = {"own_over_D": float(nn_t.median() / D),
                                       "specificity": float((nn_t / torch.cdist(own, pool).min(1).values).median()),
                                       "distinct": len(own)}
            for k, v in descs.items():
                s = v[j * S:(j + 1) * S]
                dist = torch.cdist(s, own)
                own_min, other_min = dist.min(1).values, torch.cdist(s, pool).min(1).values
                row[k] = {"own_over_D": float(own_min.median() / D),
                          "specificity": float((own_min / other_min).median()),
                          "distinct": len(set(dist.argmin(1).tolist()))}
            rows.append(row)
            if gi in image_groups:
                g = d.groups[gi]
                for w in args.guidance:
                    k = f"enricher_g{w:g}"
                    for i in range(min(3, S)):
                        m = int(n[j * S + i])
                        key = f"{g['id']}-g{w:g}-s{i}"
                        image_out[key] = d.to_raw(out[k][j * S + i, :m]).to(torch.bfloat16).contiguous()
                        near = int(torch.cdist(descs[k][j * S + i:j * S + i + 1], own).argmin())
                        image_index.append({"key": key, "group": g["id"], "base": g["base"], "k": g["k"], "rows": m,
                                            "guidance": w, "nearest_detail": g["details"][near]})
    summary = {}
    for k in ["held_out_teacher", *descs]:
        summary[k] = {"median_own_over_D": float(np.median([r[k]["own_over_D"] for r in rows])),
                      "specific_fraction": float(np.mean([r[k]["specificity"] <= 0.85 for r in rows])),
                      "median_specificity": float(np.median([r[k]["specificity"] for r in rows])),
                      "median_distinct": float(np.median([r[k]["distinct"] for r in rows]))}
    ref = summary["held_out_teacher"]
    # Gate rule from 2026-11-15: compare against a real held-out enrichment, not against regression.
    gates = {k: {"fit_within_10pct_of_teacher": v["median_own_over_D"] <= 1.1 * ref["median_own_over_D"],
                 "specific_at_least_75pct_of_teacher": v["specific_fraction"] >= 0.75 * ref["specific_fraction"],
                 "diverse": v["median_distinct"] >= 4}
             for k, v in summary.items() if k.startswith("enricher")}
    report = {"checkpoint": str(args.checkpoint), "step": step, "mode": mode, "split": args.split, "groups": len(rows),
              "samples": S, "steps": args.steps, "length_mae": float(np.mean(length_err)), "summary": summary,
              "gates": gates, "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"score-{args.split}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if image_out:
        save_file(image_out, str(args.output / f"image-samples-{args.split}.safetensors"))
        (args.output / f"image-index-{args.split}.json").write_text(json.dumps(image_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("data")
    a.add_argument("--shards", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--tag", nargs="*", default=[], help="name=curated.jsonl: tag these group ids")
    a.add_argument("--pca", type=int, default=0, help="0 = smallest of 256/384/512 with 95%% variance")
    a.add_argument("--limit-shards", type=int, default=0)
    a.add_argument("--threads", type=int, default=12)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--mode", choices=("flow", "regress"), default="flow")
    t.add_argument("--dim", type=int, default=512)
    t.add_argument("--layers", type=int, default=8)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--length-weight", type=float, default=0.1)
    t.add_argument("--ctx-drop", type=float, default=0.0, help="Fraction of flow steps trained without base rows")
    t.add_argument("--every", type=int, default=250)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--time-limit", type=float, default=0, help="Minutes; save and stop when it passes")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--threads", type=int, default=12)
    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--regression", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    s.add_argument("--max-groups", type=int, default=0)
    s.add_argument("--samples", type=int, default=8)
    s.add_argument("--steps", type=int, default=16)
    s.add_argument("--guidance", type=float, nargs="+", default=[1.0])
    s.add_argument("--others", type=int, default=64)
    s.add_argument("--chunk", type=int, default=64)
    s.add_argument("--image-groups", type=int, default=0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    {"data": data, "train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
