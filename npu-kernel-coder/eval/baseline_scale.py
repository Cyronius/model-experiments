"""Deterministic shape-port baseline: rescale A's numbers to B's shape.

    python3 baseline_scale.py ../npuasm/results/corpus.jsonl [--tag baseline-scale]

No model. For every number in build A's npuasm text, predict its value in
build B from the design's shape parameters (the numbers in the build name):

BL1 (fit): the design's other builds (same directory, name template and
core count; never B) are aligned with A line by line, numbers masked. For
each number position the simplest relation consistent with every aligned
build is chosen: constant; a*P+b in one parameter; a*log2(P)+b; a*P*Q+b or
a*P/Q+b in two. It is evaluated at B's parameters. Fractions keep the fit
exact; a non-integer or negative prediction, or disagreeing candidates,
keeps A's value.

BL0 (guess, when no other build exists): a number equal to c*P for a
changed parameter P and a "round" c (2^k, 1/2^k, 3, 5, 6, 12), or to c*P-1 or
c*P+1 (loop counts), is scaled to c*P_B (+-1). Only size-like configuration fields (lengths, wraps, strides,
repeat counts, ddr_patch offsets) and the `#` immediates of movs into
general registers in core code: never addresses, ids, negative numbers or
memory offsets (those matched by coincidence).

Both: a 32-bit immediate whose fp32 value is exactly f(P_A), f in
{P, 1/P, sqrt P, 1/sqrt P}, for a changed parameter becomes f(P_B); BL1
applies this where its fit leaves a number unchanged.

Labels and data words with leading zeros are never touched. The answer is
written as a unified diff from A, under runs/<tag>/<task>/, scored with the
same scorer as model runs, so npu_check.py works on it unchanged.
"""

from __future__ import annotations

import argparse
import collections
import struct
import difflib
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "npuasm"))

import run  # noqa: E402
import score  # noqa: E402
from make_tasks import HASH, template  # noqa: E402
from npuasm import build  # noqa: E402

NUM = re.compile(r"\.L[0-9a-f]+\b|\{[^{}]*\}|0x[0-9a-f]+|\d+")
ROUND = [Fraction(2) ** k for k in range(-6, 7)] + [Fraction(3), Fraction(5), Fraction(6), Fraction(12)]


def params(name: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", HASH.sub("", name))]


def tokens(line: str, *, with_pos: bool = False):
    """(shape, [(text, value or None)]); labels and {..} terms are opaque.
    with_pos: entries are (text, value, start, negative)."""
    out, shape, last = [], [], 0
    for m in NUM.finditer(line):
        t = m.group(0)
        shape.append(line[last:m.start()])
        last = m.end()
        if t.startswith((".L", "{")):
            v = None
        elif t.startswith("0x"):
            v = int(t, 16) if (t == "0x0" or t[2] != "0") else None
        else:
            v = int(t) if (t == "0" or t[0] != "0") else None
        neg = m.start() > 0 and line[m.start() - 1] == "-"
        out.append((t, v, m.start(), neg) if with_pos else (t, v))
    shape.append(line[last:])
    return "\0".join(shape), out


def fit(points: list[tuple[list[int], int]], pb: list[int]):
    """Predict the value at parameters pb from (params, value) points."""
    vals = {v for _, v in points}
    if len(vals) == 1:
        return points[0][1], "const"
    preds = {}
    n = len(pb)

    def check(f, name):
        xs = []
        for p, v in points:
            x = f(p)
            if x is None:
                return
            xs.append((x, Fraction(v)))
        dx = sorted({x for x, _ in xs})
        if len(dx) < 2:
            return
        y0 = next(y for x, y in xs if x == dx[0])
        y1 = next(y for x, y in xs if x == dx[-1])
        a = (y1 - y0) / (dx[-1] - dx[0])
        b = y0 - a * dx[0]
        if all(a * x + b == y for x, y in xs):
            xb = f(pb)
            if xb is not None:
                preds[name] = a * xb + b

    for k in range(n):
        check(lambda p, k=k: Fraction(p[k]), f"P{k}")
        check(lambda p, k=k: Fraction(int(math.log2(p[k]))) if p[k] > 0 and p[k] & (p[k] - 1) == 0 else None, f"log2P{k}")
    if not preds:
        for k in range(n):
            for l in range(n):
                if k < l:
                    check(lambda p, k=k, l=l: Fraction(p[k]) * p[l], f"P{k}*P{l}")
                if k != l:
                    check(lambda p, k=k, l=l: Fraction(p[k], p[l]) if p[l] else None, f"P{k}/P{l}")
    good = {v for v in preds.values()}
    if len(good) == 1:
        v = good.pop()
        if v.denominator == 1 and v >= 0:
            return int(v), "fit"
    return None, "ambiguous" if preds else "nofit"


FLOAT_FNS = [("P", lambda x: x), ("1/P", lambda x: 1 / x), ("sqrtP", math.sqrt), ("1/sqrtP", lambda x: 1 / math.sqrt(x))]


def f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def float_rescale(v: int, pa, pb) -> int | None:
    """A 32-bit word whose fp32 value is exactly f(P_A) for a changed
    parameter becomes f(P_B) (normalization constants such as 1/sqrt(h))."""
    if not (0x30000000 <= v <= 0x4F800000):  # plausible positive normal floats
        return None
    for k in range(min(len(pa), len(pb))):
        if pa[k] == pb[k] or pa[k] <= 0 or pb[k] <= 0:
            continue
        for _, g in FLOAT_FNS:
            if f32bits(g(pa[k])) == v:
                return f32bits(g(pb[k]))
    return None


def mov_immediate(line: str, start: int) -> bool:
    """Is the number at `start` the `#` immediate of a register-loading mov?"""
    a = line.rfind(";", 0, start) + 1
    slot = line[a:start].strip()
    # general registers only: address/stride registers (p, m, dj, dn, dc) hold layout, not shape
    return bool(re.match(r"^(movx|movxm|mova|mov)\s+r\d+,\s*#$", slot))


SIZE_FIELD = re.compile(r"(?:BUFFER_LENGTH|_WRAP|_STEPSIZE|REPEAT_COUNT|ITERATION_WRAP|ITERATION_STEPSIZE|plus)=$|^repeat $")


def size_field(line: str, start: int) -> bool:
    """In configuration text, scale only size-like fields: lengths, wraps,
    strides, repeat/iteration counts, ddr_patch offsets, `repeat N`. Base
    addresses, lock and BD ids are layout, not shape."""
    return bool(SIZE_FIELD.search(line[:start]))


def fmt(old: str, v: int) -> str:
    return f"{v:#x}" if old.startswith("0x") else str(v)


def rescale_bl1(a_text: dict[str, str], others: list[tuple[list[int], dict[str, str]]], pa, pb, stats) -> dict[str, str]:
    out = {}
    for f, text in a_text.items():
        lines = text.split("\n")
        toks = [tokens(l) for l in lines]
        # aligned values per (line, number index) from each other build
        points: dict[tuple[int, int], list] = collections.defaultdict(list)
        for i, (shape, ts) in enumerate(toks):
            for j, (t, v) in enumerate(ts):
                if v is not None:
                    points[(i, j)].append((pa, v))
        for px, xt in others:
            if f not in xt:
                continue
            xl = xt[f].split("\n")
            xtoks = [tokens(l) for l in xl]
            sm = difflib.SequenceMatcher(None, [s for s, _ in toks], [s for s, _ in xtoks], autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    continue
                for di in range(i2 - i1):
                    for j, ((t, v), (tx, vx)) in enumerate(zip(toks[i1 + di][1], xtoks[j1 + di][1])):
                        if v is not None and vx is not None:
                            points[(i1 + di, j)].append((px, vx))
        new_lines = []
        for i, (shape, ts) in enumerate(toks):
            pieces = shape.split("\0")
            buf = [pieces[0]]
            for j, (t, v) in enumerate(ts):
                nt = t
                if v is not None:
                    pred, kind = fit(points[(i, j)], pb)
                    stats[kind] += 1
                    if pred is not None and pred != v:
                        nt = fmt(t, pred)
                    elif t.startswith("0x"):
                        fv = float_rescale(v, pa, pb)
                        if fv is not None:
                            nt = fmt(t, fv)
                            stats["float"] += 1
                buf.append(nt)
                buf.append(pieces[j + 1])
            new_lines.append("".join(buf))
        out[f] = "\n".join(new_lines)
    return out


def rescale_bl0(a_text: dict[str, str], pa, pb, stats) -> dict[str, str]:
    changed = [k for k in range(min(len(pa), len(pb))) if pa[k] != pb[k] and pa[k] > 0]
    out = {}
    for f, text in a_text.items():
        new_lines = []
        core = f.endswith(".s")
        for line in text.split("\n"):
            shape, ts = tokens(line, with_pos=True)
            pieces = shape.split("\0")
            buf = [pieces[0]]
            for j, (t, v, start, neg) in enumerate(ts):
                nt = t
                if v is not None and t.startswith("0x"):
                    fv = float_rescale(v, pa, pb)
                    if fv is not None:
                        buf.append(fmt(t, fv))
                        buf.append(pieces[j + 1])
                        stats["float"] += 1
                        continue
                eligible = v is not None and v >= 16 and not neg and (mov_immediate(line, start) if core else size_field(line, start))
                if eligible:
                    done = False
                    for k in changed:
                        for off in (0, -1, 1):  # loop counts are often N-1
                            c = Fraction(v - off, pa[k])
                            if c in ROUND:
                                nv = c * pb[k] + off
                                if nv.denominator == 1 and nv >= 0:
                                    nt = fmt(t, int(nv))
                                    stats["scaled"] += 1
                                done = True
                                break
                        if done:
                            break
                buf.append(nt)
                buf.append(pieces[j + 1])
            new_lines.append("".join(buf))
        out[f] = "\n".join(new_lines)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--tag", default="baseline-scale")
    ap.add_argument("--max-others", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.corpus.open()]
    groups = collections.defaultdict(list)
    for r in rows:
        if r["error"] or not all(r["ok"].values()):
            continue
        b = Path(r["prj"]).parent
        groups[(str(b.parent), template(b.name), r["cores"])].append(r["prj"])
    cache: dict[str, dict[str, str]] = {}

    def text(prj: str) -> dict[str, str]:
        if prj not in cache:
            cache[prj] = build.lift(Path(prj), None)[1]
        return cache[prj]

    index = json.loads((HERE / "tasks/index.json").read_text())
    root = HERE / "runs" / a.tag
    root.mkdir(parents=True, exist_ok=True)
    for m in index:
        td = HERE / "tasks" / m["id"]
        a_text = {p.name: p.read_text() for p in (td / "a").iterdir()}
        pa, pb = params(m["a_name"]), params(m["b_name"])
        key = (str(Path(m["a_prj"]).parent.parent), m["template"], m["cores"])
        others_prj = [p for p in groups.get(key, []) if p not in (m["a_prj"], m["b_prj"])]
        others_prj.sort(key=lambda p: sum(abs(math.log2(max(x, 1)) - math.log2(max(y, 1)))
                                           for x, y in zip(params(Path(p).parent.name), pb)))
        others = [(params(Path(p).parent.name), text(p)) for p in others_prj[: a.max_others]]
        stats = collections.Counter()
        if others:
            pred = rescale_bl1(a_text, others, pa, pb, stats)
            method = "BL1"
        else:
            pred = rescale_bl0(a_text, pa, pb, stats)
            method = "BL0"
        diff = "".join("".join(difflib.unified_diff(a_text[f].splitlines(True), pred[f].splitlines(True), f"a/{f}", f"b/{f}", n=3))
                       for f in sorted(a_text))
        rd = root / m["id"]
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "answer.txt").write_text(diff)
        res = {**m, "run": {"method": method, "other_builds": len(others), "stats": dict(stats), "seconds": 0},
               "score": score.score(td, diff)}
        (rd / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        s = res["score"]
        print(f"{m['id']} {m['difficulty']:6s} {method} others={len(others):2d} exact={s['exact']} gap={s['gap_closed']} "
              f"builds={s['builds']} rules={s['rules_pass']}/{s.get('rules_pass_aligned')} bytes={s['bytes_match']}", flush=True)
    run.summarize(root)


if __name__ == "__main__":
    main()
