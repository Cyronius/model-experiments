"""Linear probe: frozen Qwen3-Embedding-0.6B hidden states -> Klein teacher conditioning rows (2026-11-16)."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

import harness

LAYERS = (4, 7, 10, 14, 18, 21, 24, 28)
SETS = {"taps3": (7, 14, 21), "all8": LAYERS}
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FIRST = 3  # rows 0-2 are template constants


def load_student(path: Path):
    model = AutoModel.from_pretrained(str(path), dtype=torch.float32).eval()
    assert model.config.num_hidden_layers >= max(LAYERS)
    return model


@torch.inference_mode()
def features(model, tok, prompts: list[str]) -> list[torch.Tensor]:
    tok.padding_side = "right"
    enc = tok([harness.chat_text(p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False)
    hs = model(**enc, output_hidden_states=True).hidden_states
    x = torch.cat([hs[k] for k in LAYERS], -1)
    return [x[i, :n] for i, n in enumerate(enc["attention_mask"].sum(1).tolist())]


def group_rows(d: dict, meta: dict, rows: torch.Tensor, n_details: int):
    """Prompts of one group and, per prompt, the target row range it owns (base: rows 3..end; detail: changed rows)."""
    base = rows[meta["base_offset"]:meta["base_offset"] + meta["base_rows"]]
    out = [(d["base"], FIRST, base[FIRST:])]
    for det in meta["details"][:n_details]:
        out.append((f"{d['base']}, {det['text']}", det["first_changed_row"],
                    rows[det["offset"]:det["offset"] + det["rows"]]))
    return out


def iter_split(args, groups, split, limit, seed):
    pick = [g for g in groups if g["split"] == split]
    random.Random(seed).shuffle(pick)
    pick = sorted(pick[:limit], key=lambda g: g["shard"])
    info = json.loads((args.data / "info.json").read_text(encoding="utf-8"))
    cur, meta, rows = None, None, None
    for g in pick:
        if g["shard"] != cur:
            cur = g["shard"]
            path = Path(info["shards"][cur])  # relative to flux-conditioning/, where the data step ran
            meta = {m["id"]: m for m in json.loads(path.read_text(encoding="utf-8"))["groups"]}
            rows = load_file(str(path.with_suffix(".safetensors")))["rows"].float()
        yield from group_rows(g, meta[g["id"]], rows, args.details)


def batches(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def collect(args, model, tok, groups, split, limit, seed, mean, std, sink):
    t0, n = time.time(), 0
    for b in batches(iter_split(args, groups, split, limit, seed), args.batch):
        feats = features(model, tok, [p for p, _, _ in b])
        for (p, start, y), x in zip(b, feats):
            assert start + len(y) == len(x), (p, start, len(y), len(x))
            sink(x[start:], (y - mean) / std)
            n += 1
        if n % (args.batch * 50) == 0:
            print(f"{split}: {n} prompts, {time.time() - t0:.0f} s", flush=True)
    print(f"{split}: {n} prompts done, {time.time() - t0:.0f} s", flush=True)


def fit(args) -> None:
    torch.set_num_threads(args.threads)
    st = load_file(str(args.data / "stats.safetensors"))
    mean, std, basis = st["mean"], st["std"], st["basis"]
    groups = json.loads((args.data / "groups.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(args.teacher / "tokenizer"))
    model = load_student(args.student)
    F, W = len(LAYERS) * model.config.hidden_size, harness.WIDTH
    acc = {"n": 0, "sx": torch.zeros(F, dtype=torch.float64), "sy": torch.zeros(W, dtype=torch.float64),
           "xx": torch.zeros(F, F, dtype=torch.float64), "xy": torch.zeros(F, W, dtype=torch.float64)}

    def add(x, y):
        x, y = x.double(), y.double()
        acc["n"] += len(x)
        acc["sx"] += x.sum(0)
        acc["sy"] += y.sum(0)
        acc["xx"] += x.T @ x
        acc["xy"] += x.T @ y

    collect(args, model, tok, groups, "train", args.train_groups, 0, mean, std, add)
    held = {"validation": ([], []), "test": ([], [])}
    for split, limit in (("validation", args.val_groups), ("test", args.test_groups)):
        collect(args, model, tok, groups, split, limit, 1, mean, std,
                lambda x, y, s=split: (held[s][0].append(x.half()), held[s][1].append(y)))
    n = acc["n"]
    mx, my = acc["sx"] / n, acc["sy"] / n
    cxx = acc["xx"] / n - torch.outer(mx, mx)
    cxy = acc["xy"] / n - torch.outer(mx, my)
    width = model.config.hidden_size
    cols = {name: torch.cat([torch.arange(width) + LAYERS.index(k) * width for k in ks]) for name, ks in SETS.items()}

    def r2(pred, y):
        return float(1 - (pred - y).square().sum() / (y - y.mean(0)).square().sum())

    report, best = {"train_rows": n, "layers": LAYERS}, None
    yv = torch.cat(held["validation"][1])
    xv = torch.cat(held["validation"][0]).double()
    for name, c in cols.items():
        a = cxx[c][:, c]
        scale = a.diagonal().mean()
        for lam in LAMBDAS:
            w = torch.linalg.solve(a + lam * scale * torch.eye(len(c), dtype=torch.float64), cxy[c])
            pred = ((xv[:, c] - mx[c]) @ w + my).float()
            score = r2(pred, yv)
            report[f"{name}_lam{lam:g}_val_r2"] = score
            print(f"{name} lambda {lam:g}: validation R2 {score:.4f}", flush=True)
            if best is None or score > best[0]:
                best = (score, name, lam, w)
    _, name, lam, w = best
    c = cols[name]
    yt = torch.cat(held["test"][1])
    pred = ((torch.cat(held["test"][0]).double()[:, c] - mx[c]) @ w + my).float()
    pca = (yt @ basis.T) @ basis
    report.update({"best": {"set": name, "lambda": lam}, "test_r2": r2(pred, yt), "test_pca256_r2": r2(pca, yt),
                   "test_rows": len(yt),
                   "test_row_cos": float(torch.nn.functional.cosine_similarity(pred, yt, -1).median()),
                   "test_pca256_row_cos": float(torch.nn.functional.cosine_similarity(pca, yt, -1).median())})
    print(json.dumps(report, indent=2), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_file({"w": w.float().contiguous(), "mx": mx[c].float(), "my": my.float(), "cols": c,
               "mean": mean, "std": std}, str(args.output / "probe.safetensors"))


def render_prep(args) -> None:
    torch.set_num_threads(args.threads)
    p = load_file(str(args.output / "probe.safetensors"))
    tok = AutoTokenizer.from_pretrained(str(args.teacher / "tokenizer"))
    model = load_student(args.student)
    targets = args.output / "targets"
    if not (targets / "manifest.json").exists():
        harness.export(argparse.Namespace(cases=args.cases, output=targets, model=args.teacher, dtype="fp32", device="cpu"))
    # One global padding vector, from earlier exports of other prompts.
    pads = []
    for f in sorted(args.padding_from.glob("*.safetensors")):
        t = load_file(str(f))
        pads.append(t["projected_conditioning"].float()[int(t["attention_mask"].sum()):])
    pad = torch.cat(pads).mean(0)
    out = args.output / "cond"
    out.mkdir(parents=True, exist_ok=True)
    for case in json.loads((targets / "manifest.json").read_text(encoding="utf-8"))["cases"]:
        f = targets / case["file"]
        t = load_file(str(f))
        real = int(t["attention_mask"].sum())
        teacher = t["projected_conditioning"].float()
        x = features(model, tok, [case["prompt"]])[0][FIRST:real]
        assert len(x) == real - FIRST
        rows = ((x[:, p["cols"]] - p["mx"]) @ p["w"] + p["my"]) * p["std"] + p["mean"]
        ctrl = teacher.clone()
        ctrl[real:] = pad
        probe = ctrl.clone()
        probe[FIRST:real] = rows
        for tag, value in (("a-teacher", teacher), ("b-meanpad", ctrl), ("c-probe", probe)):
            harness.save_prediction(f, value.to(torch.bfloat16), out / f"{case['id']}-{tag}.safetensors")
        print(case["id"], real, "rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "render-prep"):
        s = sub.add_parser(name)
        s.add_argument("--student", type=Path, required=True)
        s.add_argument("--teacher", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--threads", type=int, default=12)
    f = sub.choices["fit"]
    f.add_argument("--data", type=Path, required=True, help="enricher-v2/data (groups, stats, shard list)")
    f.add_argument("--train-groups", type=int, default=2000)
    f.add_argument("--val-groups", type=int, default=150)
    f.add_argument("--test-groups", type=int, default=300)
    f.add_argument("--details", type=int, default=3)
    f.add_argument("--batch", type=int, default=16)
    r = sub.choices["render-prep"]
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--padding-from", type=Path, required=True)
    args = parser.parse_args()
    fit(args) if args.cmd == "fit" else render_prep(args)


if __name__ == "__main__":
    main()
