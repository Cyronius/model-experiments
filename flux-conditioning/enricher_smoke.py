"""Dumb enricher smoke test (2026-11-10): toy append-form cases, a tiny rectified-flow enricher, vector gates."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

SUBJECTS = [
    "A red fox", "A vintage bicycle", "An old lighthouse", "A ceramic teapot", "A sleeping cat",
    "A wooden rowboat", "A steam locomotive", "A barn owl", "A stone bridge", "A bowl of ramen",
    "A grand piano", "A hot air balloon", "A giant tortoise", "A red telephone box", "A sailboat",
    "A cactus in a pot", "A knight in armor", "A yellow taxi", "A brown bear", "A glass bottle",
    "A windmill", "A garden gnome", "An astronaut", "A golden retriever",
]
PHRASES = [
    "at sunset", "in heavy rain", "under neon lights", "in black and white", "as a watercolor painting",
    "in a snowy forest", "on a sandy beach", "at night under a full moon", "in thick fog",
    "as a pixel art sprite", "in a field of sunflowers", "made of LEGO bricks",
    "inside a glass snow globe", "as a charcoal sketch", "on the surface of Mars", "covered in autumn leaves",
]
HELDOUT_SUBJECTS = {"A windmill", "An astronaut"}
STAGE1 = [("s1-fox-base", "A red fox in fresh snow"),
          ("s1-fox-enriched", "A red fox in fresh snow, wet fur, at dusk, cedar trees behind")]
WINDOW, WIDTH, SEED = 32, 3072, 17


def slug(text: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def build_cases() -> list[dict]:
    cases = [{"id": i, "split": "test", "category": "stage1", "prompt": p, "seed": SEED} for i, p in STAGE1]
    held = 0
    for subject in SUBJECTS:
        test_subject = subject in HELDOUT_SUBJECTS
        base_id = f"b-{slug(subject)}"
        cases.append({"id": base_id, "split": "test" if test_subject else "train", "category": "base",
                      "prompt": subject, "seed": SEED})
        held_phrases = set() if test_subject else {(2 * held) % 16, (2 * held + 1) % 16}
        held += not test_subject
        for j, phrase in enumerate(PHRASES):
            split = "test" if test_subject else "validation" if j in held_phrases else "train"
            cases.append({"id": f"e-{slug(subject)}--{slug(phrase)}", "split": split, "category": "enriched",
                          "prompt": f"{subject}, {phrase}", "seed": SEED, "base_id": base_id, "phrase": j})
    return cases


def first_change(base: dict, enriched: dict) -> int:
    """Index of the first enriched row that may differ; earlier rows must be identical in the teacher."""
    a, b = base["input_ids"], enriched["input_ids"]
    n = 0
    while a[n] == b[n]:
        n += 1
    return n


def load_pairs(targets: Path) -> list[dict]:
    manifest = json.loads((targets / "manifest.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in manifest["cases"]}
    tensors = {c["id"]: load_file(str(targets / c["file"])) for c in manifest["cases"]}
    pairs = []
    for c in manifest["cases"]:
        if c["category"] != "enriched":
            continue
        base, enr = tensors[c["base_id"]], tensors[c["id"]]
        k = first_change(base, enr)
        b_cond, e_cond = base["projected_conditioning"].float(), enr["projected_conditioning"].float()
        # Append form premise: rows before the first changed token are bit-identical.
        if not torch.equal(b_cond[:k], e_cond[:k]):
            raise ValueError(f"Prefix rows differ for {c['id']}")
        if int(enr["attention_mask"].sum()) - k > WINDOW - 8:
            raise ValueError(f"Window too small for {c['id']}")
        pairs.append({"id": c["id"], "split": c["split"], "base_id": c["base_id"], "phrase": c["phrase"],
                      "k": k, "base_len": int(base["attention_mask"].sum()),
                      "context": b_cond[:k], "window": e_cond[k:k + WINDOW], "full": e_cond,
                      "base_full": b_cond, "file": by_id[c["id"]]["file"]})
    return pairs


class Enricher(nn.Module):
    """Tiny transformer: clean base-prompt rows as context, 32 noisy window rows to denoise."""

    def __init__(self, dim: int = 256, layers: int = 4, heads: int = 4, ctx_max: int = 32, io: int = WIDTH):
        super().__init__()
        self.inp, self.ctx, self.out = nn.Linear(io, dim), nn.Linear(io, dim), nn.Linear(dim, io)
        self.pos = nn.Parameter(torch.randn(WINDOW, dim) * 0.02)
        self.ctx_pos = nn.Parameter(torch.randn(ctx_max, dim) * 0.02)
        self.kind = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(64, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(torch.arange(32, device=t.device) * (-math.log(1000.0) / 32))
        emb = torch.cat([torch.sin(t[:, None] * 1000 * freqs), torch.cos(t[:, None] * 1000 * freqs)], -1)
        h = self.inp(x) + self.pos + self.kind[0] + self.time(emb)[:, None]
        c = self.ctx(ctx) + self.ctx_pos[: ctx.shape[1]] + self.kind[1]
        seq = torch.cat([c, h], 1)
        pad = torch.cat([~ctx_mask, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)], 1)
        return self.out(self.norm(self.body(seq, src_key_padding_mask=pad)[:, ctx.shape[1]:]))


def to_io(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return x @ stats["basis"].T / stats["scale"] if "basis" in stats else x


def from_io(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return ((x * stats["scale"]) @ stats["basis"] if "basis" in stats else x) * stats["std"] + stats["mean"]


def batchify(items: list[dict], stats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = max(len(p["context"]) for p in items)
    ctx = torch.zeros(len(items), n, WIDTH)
    mask = torch.zeros(len(items), n, dtype=torch.bool)
    for i, p in enumerate(items):
        ctx[i, : len(p["context"])] = (p["context"] - stats["mean"]) / stats["std"]
        mask[i, : len(p["context"])] = True
    win = torch.stack([(p["window"] - stats["mean"]) / stats["std"] for p in items])
    return to_io(ctx, stats), mask, to_io(win, stats)


@torch.no_grad()
def sample(model: Enricher, ctx, mask, steps: int, gen: torch.Generator, flow: bool, x0: bool = False) -> torch.Tensor:
    shape = (ctx.shape[0], WINDOW, ctx.shape[-1])
    x = torch.randn(shape, generator=gen) if flow else torch.zeros(shape)
    if not flow:
        return model(x, torch.zeros(len(x)), ctx, mask)
    for i in range(steps):
        t = torch.full((len(x),), 1 - i / steps)
        out = model(x, t, ctx, mask)
        # x0 mode: velocity = (x - x0) / t; the last step lands exactly on the x0 estimate.
        v = (x - out) / t[:, None, None] if x0 else out
        x = x - v / steps
    return x


def train(args) -> None:
    torch.manual_seed(args.seed)
    pairs = load_pairs(args.targets)
    tr = [p for p in pairs if p["split"] == "train"]
    rows = torch.cat([torch.cat([p["context"][3:], p["window"]]) for p in tr])
    # Rows 0-2 are template constants with a huge norm; keep them out of the scale estimate.
    stats = {"mean": rows.mean(0), "std": rows.std(0).clamp_min(1e-3)}
    if args.pca:
        # Latent diffusion: noise lives only in the top PCA directions, so a narrow body sees all of it.
        _, sv, v = torch.linalg.svd((rows - stats["mean"]) / stats["std"], full_matrices=False)
        stats["basis"] = v[: args.pca].contiguous()
        # Whiten so every latent direction has unit variance, matching the unit noise.
        stats["scale"] = (sv[: args.pca] / (len(rows) - 1) ** 0.5) if args.whiten else torch.ones(args.pca)
    model = Enricher(io=args.pca or WIDTH)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(args.seed)
    log = []
    for step in range(args.steps):
        batch = [tr[i] for i in torch.randint(len(tr), (args.batch,), generator=gen)]
        ctx, mask, x1 = batchify(batch, stats)
        if args.flow:
            # Rectified flow: x_t = t*noise + (1-t)*data, predict velocity noise - data.
            t = torch.rand(len(batch), generator=gen)
            noise = torch.randn(x1.shape, generator=gen)
            xt = t[:, None, None] * noise + (1 - t[:, None, None]) * x1
            pred = model(xt, t, ctx, mask)
            # A 256-wide body cannot carry 3,072-wide noise, so x0 mode predicts the clean window instead.
            loss = ((pred - x1) if args.x0 else (pred - (noise - x1))).square().mean()
        else:
            loss = (model(torch.zeros_like(x1), torch.zeros(len(batch)), ctx, mask) - x1).square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss)})
            print(f"step {step} loss {float(loss):.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats, "flow": args.flow, "x0": args.x0, "args": vars(args) | {
        "targets": str(args.targets), "output": str(args.output)}, "log": log}, args.output)


def score(args) -> None:
    pairs = load_pairs(args.targets)
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = Enricher(io=len(ckpt["stats"]["basis"]) if "basis" in ckpt["stats"] else WIDTH)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats, flow, x0 = ckpt["stats"], ckpt["flow"], ckpt.get("x0", False)
    if "basis" in stats and "scale" not in stats:
        stats["scale"] = torch.ones(len(stats["basis"]))
    gen = torch.Generator().manual_seed(args.seed)
    by_base: dict[str, list[dict]] = {}
    for p in pairs:
        by_base.setdefault(p["base_id"], []).append(p)
    report, samples_out = {"checkpoint": str(args.checkpoint), "flow": flow, "x0": x0, "groups": {}}, {}
    for group in ("train", "validation", "test"):
        rows = []
        for base_id, items in sorted(by_base.items()):
            if group == "test" and items[0]["split"] != "test":
                continue
            if group != "test" and items[0]["split"] == "test":
                continue
            if group == "validation" and not any(p["split"] == "validation" for p in items):
                continue
            # Teacher windows for all 16 phrases of this subject, in raw space.
            teach = torch.stack([p["window"] for p in items]).flatten(1)
            d_teacher = torch.cdist(teach, teach)
            D = d_teacher[~torch.eye(len(items), dtype=torch.bool)].median()
            ctx, mask, _ = batchify([items[0]] * args.samples, stats)
            out = from_io(sample(model, ctx, mask, args.steps, gen, flow, x0), stats)
            d = torch.cdist(out.flatten(1), teach)
            nearest = d.argmin(1)
            mean_window = teach.mean(0, keepdim=True)
            rows.append({
                "base_id": base_id,
                "nearest_over_D": float(d.min(1).values.median() / D),
                "distinct_nearest": len(set(nearest.tolist())),
                "nearest_phrases": [items[i]["phrase"] for i in nearest.tolist()],
                "heldout_phrases": [p["phrase"] for p in items if p["split"] == "validation"],
                "mean_of_teachers_over_D": float(torch.cdist(mean_window, teach).median() / D),
            })
            samples_out[f"{group}--{base_id}"] = out.contiguous()
        report["groups"][group] = {
            "subjects": len(rows),
            "median_nearest_over_D": float(torch.tensor([r["nearest_over_D"] for r in rows]).median()),
            "median_distinct_nearest": float(torch.tensor([float(r["distinct_nearest"]) for r in rows]).median()),
            "rows": rows,
        }
        g = report["groups"][group]
        print(f"{group}: subjects={g['subjects']} nearest/D={g['median_nearest_over_D']:.3f} "
              f"distinct={g['median_distinct_nearest']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_file(samples_out, str(args.output / "samples.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--targets", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--flow", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--x0", action="store_true")
    t.add_argument("--pca", type=int, default=0, help="Diffuse in this many PCA directions (0 = full width)")
    t.add_argument("--whiten", action="store_true")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--targets", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--samples", type=int, default=16)
    s.add_argument("--steps", type=int, default=16)
    s.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "cases":
        cases = build_cases()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(x) + "\n" for x in cases), encoding="utf-8")
        print(f"{len(cases)} cases: " + ", ".join(
            f"{s}={sum(x['split'] == s and x['category'] == 'enriched' for x in cases)}"
            for s in ("train", "validation", "test")))
    elif args.command == "train":
        torch.set_num_threads(12)
        train(args)
    else:
        torch.set_num_threads(12)
        score(args)


if __name__ == "__main__":
    main()
