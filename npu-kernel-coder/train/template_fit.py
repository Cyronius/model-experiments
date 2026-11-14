"""How much of a shape port is deterministic? A no-ML template baseline for npuasm builds.

    python3 template_fit.py [--builds ~/.cache/model-experiments/nkc-train/builds] [--designs ln,gemv_q4]
                            [--mode flat|roll] [--split loo|extrapolate]

Every swept build is lifted to npuasm text (cached in data/lifted/<design>.jsonl). Each file is
parsed into a template: its lines with every number pulled out into a slot (hex and decimal
numbers, digits inside identifiers such as `k2048` or `r12`, hex label addresses such as `.L1c0`).
With --mode roll, runs of consecutive line blocks that differ only in numbers that step by a
constant per copy are rolled into one REPEAT item (count, then base and step for each slot),
innermost first, so a file that unrolls a loop a shape-dependent number of times keeps one
template. Two builds share a template when the skeletons (everything but the slot values) match.

Each slot is then fitted as an exact function of the shape parameters (the numbers in the build
name): a constant, or a*f + b for one feature f (p, p^2, log2 p, p*q, p/q, p*q*r), with a, b
rational, fitted on all training builds of the same template. The held-out file is never consulted
before scoring: the training templates are tried nearest shape first, and the first whose slots all
fit gives the prediction ("nofit" if none does); it is "exact" if the rendered text equals the real file
byte for byte. Number spelling comes from the training build: npuasm's decimal-below-1000 rule
where the number follows it, else that slot's hex/decimal and zero padding.

--split loo: leave one build out at a time. --split extrapolate: hold out every build that has
the largest value of some shape parameter, train on the rest (new models tend to be bigger).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "npuasm"))

PARAM = re.compile(r"([a-z]+)(\d+)")
TOK = re.compile(r"(?P<lab>(?<=\.L)[0-9a-f]+)|(?P<hex>-?0x[0-9a-fA-F]+)"
                 r"|(?P<dec>-?(?<![0-9A-Za-z.])\d+)|(?P<id>(?<=[A-Za-z_])\d+)")
MAX_PERIOD = 64


# ---------------------------------------------------------------- lifting

def lift_one(prj: str):
    from npuasm import build
    try:
        _, texts, _ = build.lift(Path(prj), None)
        return prj, texts
    except Exception as e:  # noqa: BLE001
        return prj, {"error": repr(e)}


def lifted(builds: Path, designs: list[str]) -> dict[str, list[dict]]:
    cache = HERE / "data" / "lifted"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in designs:
        f = cache / f"{d}.jsonl"
        have = {}
        if f.exists():
            for l in f.open():
                r = json.loads(l)
                have[r["name"]] = r
        prjs = [p for p in sorted((builds / d).glob("*/final.prj"))
                if p.parent.name not in have and (p.parent / "insts.bin").exists()]
        if prjs:
            with ProcessPoolExecutor(12) as pool, f.open("a") as w:
                for prj, texts in pool.map(lift_one, map(str, prjs)):
                    if "error" in texts:
                        print(f"  lift failed {prj}: {texts['error'][:120]}", file=sys.stderr)
                        continue
                    name = Path(prj).parent.name
                    r = {"name": name, "params": {k: int(v) for k, v in PARAM.findall(name.removeprefix("build_"))},
                         "texts": texts}
                    have[name] = r
                    w.write(json.dumps(r) + "\n")
        out[d] = sorted(have.values(), key=lambda r: r["name"])
    return out


# ---------------------------------------------------------------- templates
# An item is a line ("L", skeleton, nums, fmts) or a repeat ("R", period, children, count, steps):
# the children are the first copy's items, and copy c has each slot of the children at base + c*step.

def auto(v: int) -> str:
    return ("-" if v < 0 else "") + (str(abs(v)) if abs(v) < 1000 else hex(abs(v)))


def parse_line(line: str):
    nums, fmts, parts, last = [], [], [], 0
    for m in TOK.finditer(line):
        s, kind = m.group(), m.lastgroup
        neg = s.startswith("-")
        digits = s.lstrip("-").removeprefix("0x")
        base = 16 if kind in ("lab", "hex") else 10
        nums.append(int(digits, base) * (-1 if neg else 1))
        pad = len(digits) if len(digits) > 1 and digits[0] == "0" else 0
        if kind in ("hex", "dec") and s == auto(nums[-1]):
            fmts.append(("auto", 0, ""))  # npuasm's rule: decimal below 1000, hex above
        else:
            fmts.append(("x" if base == 16 else "d", pad, "0x" if kind == "hex" else ""))
        parts.append(line[last:m.start()])
        last = m.end()
    parts.append(line[last:])
    return ("L", "\x00".join(parts), nums, fmts)


def skel(it) -> str:
    if it[0] == "L":
        return it[1]
    return f"\x01{it[1]}{{" + "\x1e".join(skel(c) for c in it[2]) + "}"


def nums(it) -> list[int]:
    if it[0] == "L":
        return list(it[2])
    base = [n for c in it[2] for n in nums(c)]
    return [it[3]] + [x for b, s in zip(base, it[4]) for x in (b, s)]


def with_nums(it, vals: list[int]):
    """A copy of item `it` (same skeleton) carrying the slot values `vals` (consumed from the front)."""
    if it[0] == "L":
        return ("L", it[1], [vals.pop(0) for _ in it[2]], it[3])
    count = vals.pop(0)
    n = sum(len(nums(c)) for c in it[2])
    pairs = [(vals.pop(0), vals.pop(0)) for _ in range(n)]
    flat = [b for b, _ in pairs]
    children = [with_nums(c, flat) for c in it[2]]
    return ("R", it[1], children, count, [s for _, s in pairs])


def render(it) -> list[str]:
    if it[0] == "L":
        out, parts = [], it[1].split("\x00")
        for p, v, (f, pad, pre) in zip(parts, it[2], it[3]):
            out += [p, auto(v) if f == "auto" else ("-" if v < 0 else "") + pre + format(abs(v), f"0{pad}{f}" if pad else f)]
        return ["".join(out + [parts[-1]])]
    lines = []
    base = [n for c in it[2] for n in nums(c)]
    for c in range(it[3]):
        vals = [b + c * s for b, s in zip(base, it[4])]
        for ch in it[2]:
            lines += render(with_nums(ch, vals))
    return lines


def roll_pass(items):
    out, i, changed = [], 0, False
    sk = [skel(x) for x in items]
    while i < len(items):
        done = False
        for p in range(1, min(MAX_PERIOD, (len(items) - i) // 2) + 1):
            block = sk[i:i + p]
            c = 1
            while i + (c + 1) * p <= len(items) and sk[i + c * p:i + (c + 1) * p] == block:
                c += 1
            if c < 2:
                continue
            copies = [[n for x in items[i + k * p:i + (k + 1) * p] for n in nums(x)] for k in range(c)]
            steps = [b - a for a, b in zip(copies[0], copies[1])]
            good = 2
            while good < c and all(v == b + good * s for v, b, s in zip(copies[good], copies[0], steps)):
                good += 1
            out.append(("R", p, items[i:i + p], good, steps))
            i += good * p
            changed = done = True
            break
        if not done:
            out.append(items[i])
            i += 1
    return out, changed


def template(text: str, roll: bool):
    items = [parse_line(l) for l in text.split("\n")]
    while roll:
        items, roll = roll_pass(items)
    return items


# ---------------------------------------------------------------- fitting

def features(p: dict[str, int]) -> dict[str, Fraction]:
    ks = sorted(p)
    f = {}
    for k in ks:
        f[k] = Fraction(p[k])
        f[f"{k}^2"] = Fraction(p[k] ** 2)
        if p[k] > 0 and p[k] & (p[k] - 1) == 0:
            f[f"log2 {k}"] = Fraction(int(math.log2(p[k])))
    for a, b in itertools.combinations(ks, 2):
        f[f"{a}*{b}"] = Fraction(p[a] * p[b])
    for a, b in itertools.permutations(ks, 2):
        if p[b]:
            f[f"{a}/{b}"] = Fraction(p[a], p[b])
    if len(ks) >= 3:
        for c in itertools.combinations(ks, 3):
            f["*".join(c)] = Fraction(math.prod(p[x] for x in c))
    return f


def fit_slot(xs: list[dict[str, Fraction]], ys: list[int]):
    """A predictor for one slot from its values in the training builds, or None."""
    if len(set(ys)) == 1:
        return lambda fx, v=ys[0]: v
    if len(ys) < 3:  # two points fit any line; demand a third to confirm
        return None
    for name in xs[0]:
        vals = [x.get(name) for x in xs]
        if any(v is None for v in vals):
            continue
        pts = {}
        for v, y in zip(vals, ys):
            pts.setdefault(v, y)
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = list(pts.items())[:2]
        a = Fraction(y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        if all(a * v + b == y for v, y in zip(vals, ys)):
            def f(fx, a=a, b=b, n=name):
                if n not in fx:
                    return None
                r = a * fx[n] + b
                return int(r) if r.denominator == 1 else None
            return f
    return None


def predict_file(train: list[tuple[dict, list]], test_p: dict, test_text: str, roll: bool) -> str:
    """train: [(params, template)] of training builds. Returns 'nofit' | 'wrong' | 'exact'.

    The held-out file is not looked at until the end: its template is chosen from the training
    builds' templates, nearest shape first (log distance), taking the first whose slots all fit."""
    groups: dict[tuple, list] = {}
    for p, tm in train:
        groups.setdefault(tuple(skel(x) for x in tm), []).append((p, tm))

    def dist(p):
        return sum(abs(math.log2(max(p[k], 1)) - math.log2(max(test_p[k], 1))) for k in test_p)
    fx = features(test_p)
    for g in sorted(groups.values(), key=lambda g: min(dist(p) for p, _ in g)):
        vecs = [[n for x in tm for n in nums(x)] for _, tm in g]
        xs = [features(p) for p, _ in g]
        pred = []
        for i in range(len(vecs[0])):
            f = fit_slot(xs, [v[i] for v in vecs])
            v = f(fx) if f else None
            if v is None:
                break
            pred.append(v)
        else:
            out = [with_nums(x, pred) for x in g[0][1]]
            return "exact" if "\n".join(l for x in out for l in render(x)) == test_text else "wrong"
    return "nofit"


def splits(rows: list[dict], how: str):
    if how == "loo":
        for i in range(len(rows)):
            yield rows[:i] + rows[i + 1:], [rows[i]]
    else:
        top = {k: max(r["params"][k] for r in rows) for k in rows[0]["params"]}
        test = [r for r in rows if any(r["params"][k] == v for k, v in top.items())]
        yield [r for r in rows if r not in test], test


def run_design(args):
    d, rows, roll, how = args
    res = defaultdict(lambda: defaultdict(int))
    cache = {}

    def tm(r, fn):
        k = (r["name"], fn)
        if k not in cache:
            cache[k] = template(r["texts"][fn], roll)
        return cache[k]
    for train, test in splits(rows, how):
        for r in test:
            for fn, text in r["texts"].items():
                kind = "core" if fn.endswith(".s") else "cdo/txn"
                tr = [(o["params"], tm(o, fn)) for o in train if fn in o["texts"]]
                res[kind][predict_file(tr, r["params"], text, roll)] += 1
    return d, {k: dict(v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", type=Path, default=Path.home() / ".cache/model-experiments/nkc-train/builds")
    ap.add_argument("--designs", default="")
    ap.add_argument("--mode", choices=("flat", "roll"), default="roll")
    ap.add_argument("--split", choices=("loo", "extrapolate"), default="loo")
    a = ap.parse_args()
    designs = a.designs.split(",") if a.designs else sorted(p.name for p in a.builds.iterdir() if p.is_dir())
    data = lifted(a.builds, designs)
    print(f"mode {a.mode}, split {a.split}")
    print(f"{'design':16s} {'builds':>6s} {'kind':8s} {'files':>6s} {'exact':>7s}")
    jobs = [(d, data[d], a.mode == "roll", a.split) for d in designs if len(data[d]) >= 6]
    with ProcessPoolExecutor(8) as pool:
        for d, res in pool.map(run_design, jobs):
            for kind, c in sorted(res.items()):
                n = sum(c.values())
                print(f"{d:16s} {len(data[d]):6d} {kind:8s} {n:6d} {c.get('exact', 0) / n:7.0%}   {c}",
                      flush=True)


if __name__ == "__main__":
    main()
