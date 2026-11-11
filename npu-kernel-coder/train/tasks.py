"""Deterministic npuasm skill tasks for a small post-trained model, from the lifted pool (pool.py).

    python3 tasks.py [--pool data/pool] [--out data/tasks] [--max-tokens 8192]

Every answer is computed, not written by a model, so each task is exact-match gradable:
- enc     bundle text -> bytes (npuasm-core's encoder), 8 bundles per example
- dec     bytes -> bundle text (the decoder)
- legal   real and perturbed bundles -> `ok <hex>` or the assembler's reason (VLIW slot rules)
- fim     a masked span of 1-16 lines in a core/CDO/txn file, with the lines around it
- unroll  a `repeat` block -> its expansion (loops.unfold);  fold: the reverse (loops.fold)
- regs    declarative CDO/txn lines -> raw register writes (decl/txndecl -> cdo/txn raw form)
- port    one file of build A at shape a -> unified diff to the same file at shape b; each
          is tagged with whether the template inducer (template_fit.py) already gets it exact;
          port_window: the same for a ~160-line window around a change, when the file is too long

Held out (val.jsonl / test.jsonl, never in train.jsonl):
- val: every build of ~10% of design families (all task kinds drawn from those builds);
- test: in each family with a numbered sweep (>= 6 builds), the builds at the largest value of
  the parameter with the most distinct values (extrapolation), and a random ~8% of the rest;
- bundles: ~2% of distinct bundles (by hash) for enc/dec/legal.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import gzip
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "npuasm"))
sys.path.insert(0, str(HERE.parent / "eval"))

from npuasm import cdo, decl, loops, txn, txndecl  # noqa: E402
from npuasm.core import core_bin  # noqa: E402
from make_tasks import HASH  # noqa: E402
import template_fit as tf  # noqa: E402

TOKENIZER = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/tokenizer.json"), None)
SYSTEM = "You are an expert in AMD XDNA2 (AIE2P) NPU programs written in npuasm. Answer with exactly what is asked."
PARAM = re.compile(r"([a-z]+)(\d+)")
REG = re.compile(r"\b(r|p|m|dj|dn|dc|s|wl|wh|x|y|bml|bmh|cml|cmh|amll|amlh|amhl|amhh|lc|ls|le|q|e|cm|bm|dm|crsat)(\d+)\b")
IMM = re.compile(r"#(-?0x[0-9a-f]+|-?\d+)")


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def params(name: str) -> dict[str, int]:
    """Shape parameters from a build name: `build_n2048_k1024_9fbe918c` -> {n: 2048, k: 1024}."""
    out = {}
    for part in HASH.sub("", name).split("_"):
        m = re.fullmatch(r"([a-z]+)(\d+)", part)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


# ---------------------------------------------------------------- held-out builds

def assign(builds: list[dict], rng: random.Random, val_share: float) -> dict[str, str]:
    """prj -> train | val | test."""
    fams = sorted({b["design"] for b in builds})
    rng.shuffle(fams)
    val = set(fams[: max(1, round(len(fams) * val_share))])
    out = {}
    groups = collections.defaultdict(list)
    for b in builds:
        if b["design"] in val:
            out[b["prj"]] = "val"
        else:
            groups[(b["design"], b["template"], b["cores"])].append(b)
    for g in groups.values():
        ps = [params(b["name"]) for b in g]
        keys = set.intersection(*(set(p) for p in ps)) if ps else set()
        if len(g) >= 6 and keys:
            k = max(sorted(keys), key=lambda k: len({p[k] for p in ps}))
            top = max(p[k] for p in ps)
            for b, p in zip(g, ps):
                out[b["prj"]] = "test" if p[k] == top or h(b["prj"]) % 12 == 0 else "train"
        else:
            for b in g:
                out[b["prj"]] = "train"
    return out


# ---------------------------------------------------------------- bundle tasks

def assemble(lines: list[str]) -> list[str]:
    r = subprocess.run([str(core_bin()), "bundles"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def perturb(text: str, pool_slots: list[str], regs_seen: list[str], rng: random.Random) -> str:
    kind = rng.random()
    slots = [s.strip() for s in text.split(";") if s.strip()]
    if kind < 0.4:  # recombine slots of different bundles
        extra = rng.sample(pool_slots, rng.randint(1, 2))
        slots = rng.sample(slots, rng.randint(1, len(slots))) + extra
        rng.shuffle(slots)
        return "; ".join(slots)
    if kind < 0.75:  # swap one register for another (often another class)
        ms = [(i, m) for i, s in enumerate(slots) for m in REG.finditer(s)]
        if ms:
            i, m = rng.choice(ms)
            s = slots[i]
            slots[i] = s[: m.start()] + rng.choice(regs_seen) + s[m.end():]
            return "; ".join(slots)
    ms = [(i, m) for i, s in enumerate(slots) for m in IMM.finditer(s)]
    if ms:  # change an immediate, sometimes out of range
        i, m = rng.choice(ms)
        v = rng.choice([rng.randint(-8, 8), 1 << rng.randint(4, 20), rng.randint(0, 1 << 24)])
        s = slots[i]
        slots[i] = s[: m.start()] + "#" + (hex(v) if v >= 1000 or v < -1000 else str(v)) + s[m.end():]
        return "; ".join(slots)
    return "; ".join(rng.sample(slots, len(slots)))  # reorder slots


def bundle_tasks(pool: Path, rng: random.Random, n_enc: int, n_legal: int) -> list[dict]:
    rows = []
    with gzip.open(pool / "bundles.tsv.gz", "rt") as fh:
        for line in fh:
            hx, t, n = line.rstrip("\n").split("\t")
            rows.append((hx, t, int(n)))
    split = lambda hx: "test" if h(hx) % 50 == 0 else "train"  # noqa: E731
    by = collections.defaultdict(list)
    for r in rows:
        by[split(r[0])].append(r)
    out = []
    slots_all = [s.strip() for _, t, _ in rows[:: max(1, len(rows) // 20000)] for s in t.split(";") if s.strip()]
    regs_seen = sorted({m.group(0) for s in slots_all for m in REG.finditer(s)})
    for sp, rs in by.items():
        scale = 1 if sp == "train" else 0.02
        for _ in range(int(n_enc * scale)):
            b = rng.sample(rs, 8)
            enc = "\n".join(t for _, t, _ in b)
            out.append(task("enc", sp, "Encode each AIE2P bundle as hex bytes (one line each, lowest address byte first):\n" + enc,
                            "\n".join(hx for hx, _, _ in b)))
            b = rng.sample(rs, 8)
            out.append(task("dec", sp, "Decode each line of AIE2P bundle bytes to npuasm text:\n" + "\n".join(hx for hx, _, _ in b),
                            "\n".join(t for _, t, _ in b)))
        cands = []
        for _ in range(int(n_legal * scale)):
            b = rng.sample(rs, 8)
            cands.append([t if rng.random() < 0.35 else perturb(t, slots_all, regs_seen, rng) for _, t, _ in b])
        flat = [t for c in cands for t in c]
        res = assemble(flat) if flat else []
        k = 0
        for c in cands:
            ans = res[k: k + len(c)]
            k += len(c)
            ans = [a.replace("\t", " ") for a in ans]
            out.append(task("legal", sp, "For each AIE2P bundle, answer `ok <hex>` with its encoding, or `err <reason>` if no "
                                          "encoding exists:\n" + "\n".join(c), "\n".join(ans)))
    return out


# ---------------------------------------------------------------- text tasks

def task(kind: str, split: str, user: str, answer: str, **meta) -> dict:
    return {"kind": kind, "split": split, "messages": [{"role": "system", "content": SYSTEM},
                                                         {"role": "user", "content": user},
                                                         {"role": "assistant", "content": answer}], "meta": meta}


def fim(b: dict, sp: str, rng: random.Random, n: int) -> list[dict]:
    out = []
    files = [f for f in b["texts"] if b["texts"][f].count("\n") > 8]
    weights = [3 if f.endswith(".s") else 1 for f in files]
    for _ in range(n if files else 0):
        f = rng.choices(files, weights)[0]
        lines = b["texts"][f].splitlines()
        k = rng.choice([1, 1, 2, 3, 4, 6, 8, 12, 16])
        if len(lines) <= k + 4:
            continue
        i = rng.randrange(1, len(lines) - k)
        pre, mid, post = lines[max(0, i - 120): i], lines[i: i + k], lines[i + k: i + k + 40]
        user = (f"File `{f}` of build `{b['name']}` ({b['design']}). Write the {k} missing line(s) at <MISSING>.\n"
                "```\n" + "\n".join(pre + ["<MISSING>"] + post) + "\n```")
        out.append(task("fim", sp, user, "\n".join(mid), design=b["design"], file=f))
    return out


def blocks(lines: list[str]):
    """(start, end) of each outermost repeat block."""
    k = 0
    while k < len(lines):
        if loops.HEAD.match(lines[k]):
            d, j = 1, k + 1
            while d and j < len(lines):
                d += 1 if loops.HEAD.match(lines[j]) else -1 if lines[j] == "}" else 0
                j += 1
            yield k, j
            k = j
        else:
            k += 1


def loop_tasks(b: dict, sp: str, rng: random.Random, n: int) -> list[dict]:
    out = []
    for f, text in b["texts"].items():
        if f.endswith(".s"):
            continue
        lines = text.splitlines()
        bl = list(blocks(lines))
        for s, e in rng.sample(bl, min(n, len(bl))):
            folded = lines[s:e]
            flat = loops.unfold(folded)
            if len(flat) > 160:
                continue
            out.append(task("unroll", sp, f"Expand the repeat loops in this snippet of `{f}` (no repeat lines in the answer):\n"
                            + "\n".join(folded), "\n".join(flat), design=b["design"], file=f))
            out.append(task("fold", sp, f"Fold the unrolled runs in this snippet of `{f}` into npuasm repeat loops "
                            "(3+ copies whose varying numbers step by a constant; nest as i, j, k innermost first):\n"
                            + "\n".join(flat), "\n".join(loops.fold(flat)), design=b["design"], file=f))
    return out


TILE = re.compile(r"^@c\d+r\d+(\s|$)")


def regs_tasks(b: dict, sp: str, rng: random.Random, n: int) -> list[dict]:
    out = []
    for f, text in b["texts"].items():
        if not (f.endswith(".cdo") or f == "insts.txn"):
            continue
        lines = decl._expand_segments("\n".join(loops.unfold(text.splitlines())))
        head = lines[0] if f == "insts.txn" else None
        body = lines[1:] if head else lines
        for _ in range(n):
            if len(body) < 4:
                break
            i = rng.randrange(0, len(body) - 1)
            k = rng.randint(1, 6)
            tile = next((body[j] for j in range(i, -1, -1) if TILE.match(body[j])), None)
            if tile is None:
                continue
            snip = [tile.split()[0]] + [l for l in body[i: i + k] if not TILE.match(l)]
            if len(snip) < 2 or any(l.startswith("dmawrite") and "@" in l for l in snip):
                continue
            try:
                if head:
                    hd, ops = txndecl.from_text("\n".join([head] + snip))
                    raw = txn.to_text(hd, ops).splitlines()[1:]
                else:
                    raw = cdo.to_text(decl.from_text("\n".join(snip))).splitlines()
            except Exception:  # noqa: BLE001
                continue
            if not raw or len(raw) > 40:
                continue
            out.append(task("regs", sp, f"Lower these `{f}` lines to raw register writes (`cCrR+offset`, values in hex):\n"
                            + "\n".join(snip), "\n".join(raw), design=b["design"], file=f))
    return out


# ---------------------------------------------------------------- ports

def unified(a: str, b: str, f: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{f}", f"b/{f}", n=3))


def windows(a: str, b: str, rng: random.Random, n: int, size: int = 160):
    """Up to n line windows [i, j) of a, each around a changed region, with the matching [k, l) of b.
    Window edges sit inside unchanged runs, so the mapping between a and b is unambiguous."""
    al, bl = a.splitlines(), b.splitlines()
    ops = difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes()
    eq = [(i1, i2, j1) for t, i1, i2, j1, _ in ops if t == "equal"]

    def to_b(i):  # a line index inside an equal run -> b index, else None
        for i1, i2, j1 in eq:
            if i1 <= i < i2:
                return j1 + i - i1
        return None
    changed = [(i1, i2) for t, i1, i2, _, _ in ops if t != "equal"]
    out = []
    for i1, i2 in rng.sample(changed, min(n, len(changed))):
        c = (i1 + i2) // 2
        i, j = max(0, c - size // 2), min(len(al), c + size // 2)
        while i > 0 and to_b(i) is None:
            i -= 1
        while j < len(al) and to_b(j - 1) is None:
            j += 1
        k = to_b(i) if i > 0 else 0
        l = to_b(j - 1) + 1 if j < len(al) else len(bl)
        if k is None or l is None or j - i > 3 * size:
            continue
        out.append(((i, j), (k, l)))
    return out


def port_tasks(builds: list[dict], where: dict[str, str], rng: random.Random, per_target: int,
               per_group: int, window_chars: int = 30000) -> list[dict]:
    groups = collections.defaultdict(list)
    for b in builds:
        if params(b["name"]):
            groups[(b["design"], b["template"], b["cores"])].append(b)
    out = []
    for (design, tmpl, _), g in groups.items():
        if len(g) < 3:
            continue
        ps = {b["prj"]: params(b["name"]) for b in g}
        keys = set.intersection(*(set(p) for p in ps.values()))
        if not keys:
            continue
        train_b = [b for b in g if where[b["prj"]] == "train"]
        # Template inducer's verdict for each (target build, file), fitted on training builds only
        # (minus the target itself), so `template_exact` says whether a model adds anything there.
        tm_cache = {}

        def tmpl_of(b, f):
            key = (b["prj"], f)
            if key not in tm_cache:
                tm_cache[key] = tf.template(b["texts"][f], True)
            return tm_cache[key]

        def dist(p, q):
            return sum(abs(math.log2(max(p[k], 1)) - math.log2(max(q[k], 1))) for k in keys)

        mine = []
        for tgt in g:
            sp = where[tgt["prj"]]
            srcs = [b for b in train_b if b is not tgt and set(b["texts"]) == set(tgt["texts"])]
            if not srcs:
                continue
            srcs.sort(key=lambda b: dist(ps[b["prj"]], ps[tgt["prj"]]))
            picks = srcs[:1] + rng.sample(srcs[1:6], min(per_target - 1, len(srcs[1:6])))
            for f in tgt["texts"]:
                fit = [b for b in train_b if b is not tgt and f in b["texts"]]
                verdict = "nofit"
                if len(fit) >= 3:
                    try:
                        verdict = tf.predict_file([({k: ps[b["prj"]][k] for k in keys}, tmpl_of(b, f)) for b in fit],
                                                  {k: ps[tgt["prj"]][k] for k in keys}, tgt["texts"][f], True)
                    except Exception:  # noqa: BLE001
                        verdict = "error"
                for src in (picks if sp == "train" else picks[:1]):
                    a, bt = src["texts"][f], tgt["texts"][f]
                    if a == bt:
                        continue
                    pa = ", ".join(f"{k}={ps[src['prj']][k]}" for k in sorted(keys))
                    pb = ", ".join(f"{k}={ps[tgt['prj']][k]}" for k in sorted(keys))
                    meta = dict(design=design, file=f, template_exact=verdict == "exact", template=verdict,
                                src=src["name"], dst=tgt["name"])
                    if len(a) + 2 * len(unified(a, bt, f)) <= window_chars:
                        user = (f"Design `{design}` ({tmpl}). Port file `{f}` from the build at {pa} (`{src['name']}`) to "
                                f"{pb}. Answer with a unified diff (3 context lines) from the file below.\n```\n{a}```")
                        mine.append(task("port", sp, user, unified(a, bt, f), **meta))
                        continue
                    # Too long whole: port windows of it instead, each around a changed region.
                    for (i, j), (k, l) in windows(a, bt, rng, 2 if sp == "train" else 1):
                        wa = "\n".join(a.splitlines()[i:j]) + "\n"
                        wb = "\n".join(bt.splitlines()[k:l]) + "\n"
                        user = (f"Design `{design}` ({tmpl}). Port lines {i + 1}-{j} of file `{f}` from the build at {pa} "
                                f"(`{src['name']}`) to {pb}. Answer with a unified diff (3 context lines) from the "
                                f"lines below.\n```\n{wa}```")
                        mine.append(task("port_window", sp, user, unified(wa, wb, f), **meta, lines=[i, j]))
        # Many-core designs repeat near-identical per-core ports: cap each group, per split.
        for sp in ("train", "val", "test"):
            es = [e for e in mine if e["split"] == sp]
            out += rng.sample(es, min(len(es), per_group if sp == "train" else max(8, per_group // 4)))
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=HERE / "data/pool")
    ap.add_argument("--out", type=Path, default=HERE / "data/tasks")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--n-enc", type=int, default=6000)
    ap.add_argument("--n-legal", type=int, default=6000)
    ap.add_argument("--fim-per-build", type=int, default=6)
    ap.add_argument("--loops-per-file", type=int, default=2)
    ap.add_argument("--regs-per-file", type=int, default=2)
    ap.add_argument("--ports-per-target", type=int, default=2)
    ap.add_argument("--ports-per-group", type=int, default=160)
    ap.add_argument("--port-max-tokens", type=int, default=12288)
    ap.add_argument("--easy-port-keep", type=float, default=0.25, help="share of template-solved train ports kept")
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))

    with gzip.open(a.pool / "builds.jsonl.gz", "rt") as fh:
        builds = [json.loads(l) for l in fh]
    where = assign(builds, rng, a.val_share)

    examples = bundle_tasks(a.pool, rng, a.n_enc, a.n_legal)
    for b in builds:
        sp = where[b["prj"]]
        examples += fim(b, sp, rng, a.fim_per_build)
        examples += loop_tasks(b, sp, rng, a.loops_per_file)
        examples += regs_tasks(b, sp, rng, a.regs_per_file)
    examples += port_tasks(builds, where, rng, a.ports_per_target, a.ports_per_group)

    drop = collections.Counter()
    kept = collections.defaultdict(list)
    for e in examples:
        if e["kind"].startswith("port") and e["split"] == "train" and e["meta"]["template_exact"] and rng.random() > a.easy_port_keep:
            drop["port solved by template"] += 1
            continue
        n = sum(len(tok.encode(m["content"], add_special_tokens=False).ids) for m in e["messages"]) + 16
        cap = a.port_max_tokens if e["kind"].startswith("port") else a.max_tokens
        if n > cap:
            drop[f"{e['kind']} over {cap} tokens"] += 1
            continue
        e["meta"]["tokens"] = n
        e["id"] = e["kind"][0].upper() + hashlib.sha256(json.dumps(e["messages"]).encode()).hexdigest()[:11]
        kept[e["split"]].append(e)

    a.out.mkdir(parents=True, exist_ok=True)
    stats = {"dropped": dict(drop), "splits": {}}
    for sp, es in kept.items():
        rng.shuffle(es)
        with (a.out / f"{sp}.jsonl").open("w") as fh:
            for e in es:
                fh.write(json.dumps(e) + "\n")
        c = collections.Counter(e["kind"] for e in es)
        t = collections.Counter()
        for e in es:
            t[e["kind"]] += e["meta"]["tokens"]
        stats["splits"][sp] = {k: {"examples": c[k], "tokens": t[k]} for k in sorted(c)}
        if sp != "train":
            ports = [e for e in es if e["kind"].startswith("port")]
            stats["splits"][sp]["port_template_exact"] = sum(e["meta"]["template_exact"] for e in ports)
    stats["builds"] = dict(collections.Counter(where.values()))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
