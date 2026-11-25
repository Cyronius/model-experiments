"""OCR-cleanup data v2: fineweb-edu text, line breaks in the character set, line-wrap damage.

Changes from gen_ocr.py:
- Text: fineweb-edu (one 3.5B-character shard; row groups 0-99 train, 725 test), not wikitext-2.
- Characters: printable ASCII plus "\\n". Input lines are wrapped at a random width; a wrap at a word
  boundary should become a space, a hyphenated wrap inside a word ("be-\\ngin") should disappear.
- Same per-character alignment: each input character emits 0-2 output characters.

  python gen_ocr2.py      # writes cache/ocr2/{train,test,heavy}.npz
"""
import multiprocessing as mp
import os
import re
import unicodedata

import numpy as np

from gen_ocr import MERGE, NOISE, SPLIT, SUB

L = 128
CLEAN = 100
CHARS = [chr(i) for i in range(32, 127)] + ["\n"]
CI = {c: i for i, c in enumerate(CHARS)}
EMPTY = len(CHARS)
PAD = len(CHARS)
SHARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "text", "fineweb-edu-000.parquet")
ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       "…": "...", " ": " "})


def load_text(groups):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(SHARD)
    out = []
    for g in groups:
        for s in f.read_row_group(g, columns=["text"]).column("text").to_pylist():
            s = unicodedata.normalize("NFKC", s.translate(ASCII))
            s = re.sub(r"\s+", " ", s).strip()  # flatten: line breaks are added by the corruption
            s = "".join(c for c in s if 32 <= ord(c) < 127)
            if len(s) > 2 * CLEAN:
                out.append(s)
    return " ".join(out)


def corrupt(clean, rate, wrap, rng):
    """Returns (input chars, outputs per input char, damaged flags). wrap: line width or 0."""
    inp, outs = [], []
    col = 0
    i = 0
    while i < len(clean):
        c = clean[i]
        if wrap and col >= wrap and c.isalpha() and i > 0 and clean[i - 1].isalpha() and rng.random() < 0.3:
            inp += ["-", "\n"]; outs += ["", ""]; col = 0       # hyphenated wrap inside a word
        if wrap and col >= wrap and c == " ":
            inp.append("\n"); outs.append(" "); col = 0; i += 1  # wrap at a word boundary
            continue
        if rng.random() >= rate:
            inp.append(c); outs.append(c); i += 1; col += 1
            continue
        ev = rng.choice(["sub", "split", "merge", "del", "ins", "case", "hyph"], p=[.35, .1, .1, .15, .15, .05, .1])
        if ev == "sub" and c in SUB:
            inp.append(rng.choice(list(SUB[c]))); outs.append(c); i += 1
        elif ev == "split" and c in SPLIT:
            a, b = SPLIT[c]
            inp += [a, b]; outs += [c, ""]; i += 1
        elif ev == "merge" and clean[i:i + 2] in MERGE:
            inp.append(MERGE[clean[i:i + 2]]); outs.append(clean[i:i + 2]); i += 2
        elif ev == "del" and outs and len(outs[-1]) == 1:
            outs[-1] += c; i += 1
        elif ev == "ins":
            inp.append(" " if rng.random() < 0.4 else rng.choice(list(NOISE))); outs.append("")
        elif ev == "case" and c.isalpha():
            inp.append(c.swapcase()); outs.append(c); i += 1
        elif ev == "hyph" and c.isalpha() and i > 0 and clean[i - 1].isalpha():
            inp += ["-", " "]; outs += ["", ""]
        else:
            inp.append(c); outs.append(c); i += 1
        col += 1
    damaged = [o != ch for ch, o in zip(inp, outs)]
    return inp, outs, damaged


def sample(text, rng, lo, hi, clean_p):
    while True:
        s = int(rng.integers(0, len(text) - 2 * CLEAN))
        s = text.find(" ", s) + 1
        e = text.rfind(" ", s, s + CLEAN + 1)
        clean = text[s:e if e > s else s + CLEAN]
        rate = 0.0 if rng.random() < clean_p else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        wrap = int(rng.integers(30, 90)) if rng.random() < 0.5 else 0
        inp, outs, dmg = corrupt(clean, rate, wrap, rng)
        if len(inp) <= L:
            assert "".join(outs) == clean
            return inp, outs, dmg


def work(job):
    groups, seed, n, lo, hi, clean_p = job
    rng = np.random.default_rng(seed)
    text = load_text(groups)
    x = np.full((n, L), PAD, np.int16)
    y = np.full((n, L, 2), EMPTY, np.int16)
    dmg = np.zeros((n, L), bool)
    pad = np.ones((n, L), bool)
    for k in range(n):
        inp, outs, d = sample(text, rng, lo, hi, clean_p)
        m = len(inp)
        x[k, :m] = [CI[c] for c in inp]
        for j, o in enumerate(outs):
            for q, ch in enumerate(o):
                y[k, j, q] = CI[ch]
        dmg[k, :m] = d
        pad[k, :m] = False
    return {"x": x, "y": y, "dmg": dmg, "pad": pad}


def build(name, groups, n, seed, lo=0.01, hi=0.15, clean_p=0.15, procs=20, per=20000):
    jobs = [([groups[i % len(groups)]], seed * 100000 + i,
             min(per, n - i * per), lo, hi, clean_p) for i in range((n + per - 1) // per)]
    with mp.Pool(procs) as pool:
        parts = pool.map(work, jobs)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs("cache/ocr2", exist_ok=True)
    np.savez(f"cache/ocr2/{name}.npz", **data)
    print(f"{name}: {len(data['x'])} samples, damaged frac {data['dmg'][~data['pad']].mean():.3f}", flush=True)


def decode(slots):
    return "".join(CHARS[c] for c in np.asarray(slots).reshape(-1) if c < EMPTY)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rng = np.random.default_rng(0)
    text = load_text([725])
    for r in (0.05, 0.15):
        inp, outs, d = sample(text, rng, r, r * 1.0001, 0.0)
        print(f"rate {r}:\n  in : {''.join(inp)!r}\n  out: {''.join(outs)!r}")
    build("test", [725], 4096, 1, per=4096)
    build("heavy", [725], 4096, 2, lo=0.15, hi=0.3, clean_p=0.0, per=4096)
    build("train", list(range(100)), 2_000_000, 3)
