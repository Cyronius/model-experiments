"""OCR-cleanup data: wikitext-2 windows with OCR-style corruption, aligned per input character.

Each input character maps to an output of 0-2 characters; concatenating the outputs gives the clean
text exactly. That lets an encoder (no autoregressive decoder) fix insertions, deletions and
many-to-one confusions such as rn -> m.

  python gen_ocr.py     # writes cache/ocr/{train,test,heavy}.npz
"""
import glob
import multiprocessing as mp
import os
import re

import numpy as np

L = 128           # input length (padded)
CLEAN = 100       # target clean window length (approx)
CHARS = [chr(i) for i in range(32, 127)]
CI = {c: i for i, c in enumerate(CHARS)}
EMPTY = len(CHARS)      # output slot class for "no character"
PAD = len(CHARS)        # input pad id (inputs never use EMPTY)
SUB = {"l": "1I|", "I": "l1|", "1": "lI", "O": "0", "o": "0ce", "0": "O", "S": "5", "5": "S", "B": "8", "8": "B",
       "e": "co", "c": "e", "a": "o", "n": "u", "u": "n", "h": "b", "b": "h", "i": "l", "t": "f", "f": "t",
       "g": "q", "q": "g", ".": ",", ",": ".", "'": "`", ";": ":", ":": ";", "Z": "2", "y": "v", "v": "y"}
SPLIT = {"m": "rn", "w": "vv", "d": "cl", "h": "li"}           # one clean char read as two
MERGE = {"rn": "m", "cl": "d", "vv": "w", "li": "h"}           # two clean chars read as one
NOISE = ".,'`~^-|:"


def load_text(split):
    path = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1/*/*/wikitext-{split}.arrow"))[0]
    import pyarrow as pa
    with pa.memory_map(path) as f:
        try:
            t = pa.ipc.open_stream(f).read_all()
        except pa.ArrowInvalid:
            t = pa.ipc.open_file(f).read_all()
    lines = []
    for s in t.column("text").to_pylist():
        s = s.strip()
        if not s or s.startswith("="):
            continue
        s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        s = re.sub(r" ([.,;:!?)'])", r"\1", s).replace("( ", "(")
        if all(32 <= ord(ch) < 127 for ch in s):
            lines.append(s)
    return " ".join(lines)


def corrupt(clean, rate, rng):
    """Returns (inp chars, outs per input char, damaged flags)."""
    inp, outs = [], []
    i = 0
    while i < len(clean):
        c = clean[i]
        if rng.random() >= rate:
            inp.append(c); outs.append(c); i += 1
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
            outs[-1] += c; i += 1                      # previous input char also emits the dropped one
        elif ev == "ins":
            inp.append(" " if rng.random() < 0.4 else rng.choice(list(NOISE))); outs.append("")
        elif ev == "case" and c.isalpha():
            inp.append(c.swapcase()); outs.append(c); i += 1
        elif ev == "hyph" and c.isalpha() and i > 0 and clean[i - 1].isalpha():
            inp += ["-", " "]; outs += ["", ""]
        else:
            inp.append(c); outs.append(c); i += 1
    damaged = [o != ch for ch, o in zip(inp, outs)]
    return inp, outs, damaged


def sample(text, rng, lo, hi, clean_p):
    while True:
        s = int(rng.integers(0, len(text) - 2 * CLEAN))
        s = text.find(" ", s) + 1
        e = text.rfind(" ", s, s + CLEAN + 1)
        clean = text[s:e if e > s else s + CLEAN]
        rate = 0.0 if rng.random() < clean_p else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        inp, outs, dmg = corrupt(clean, rate, rng)
        if len(inp) <= L:
            assert "".join(outs) == clean
            return inp, outs, dmg, rate


def work(job):
    split, seed, n, lo, hi, clean_p = job
    rng = np.random.default_rng(seed)
    text = load_text(split)
    x = np.full((n, L), PAD, np.int16)
    y = np.full((n, L, 2), EMPTY, np.int16)
    dmg = np.zeros((n, L), bool)
    pad = np.ones((n, L), bool)
    rate = np.zeros(n, np.float32)
    for k in range(n):
        inp, outs, d, r = sample(text, rng, lo, hi, clean_p)
        m = len(inp)
        x[k, :m] = [CI[c] for c in inp]
        for j, o in enumerate(outs):
            for q, ch in enumerate(o):
                y[k, j, q] = CI[ch]
        dmg[k, :m] = d
        pad[k, :m] = False
        rate[k] = r
    return {"x": x, "y": y, "dmg": dmg, "pad": pad, "rate": rate}


def build(name, split, n, seed, lo=0.01, hi=0.15, clean_p=0.15, procs=24):
    chunk = 2000
    jobs = [(split, seed * 100000 + i, min(chunk, n - i * chunk), lo, hi, clean_p) for i in range((n + chunk - 1) // chunk)]
    with mp.Pool(procs) as pool:
        parts = pool.map(work, jobs)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs("cache/ocr", exist_ok=True)
    np.savez(f"cache/ocr/{name}.npz", **data)
    print(f"{name}: {len(data['x'])} samples, damaged frac {data['dmg'][~data['pad']].mean():.3f}", flush=True)


def decode(slots):
    """slots [L,2] class ids -> string."""
    return "".join(CHARS[c] for c in np.asarray(slots).reshape(-1) if c < EMPTY)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rng = np.random.default_rng(0)
    text = load_text("train")
    print(f"train text {len(text) / 1e6:.1f}M chars")
    for r in (0.05, 0.15):
        inp, outs, d, _ = sample(text, rng, r, r * 1.0001, 0.0)
        print(f"rate {r}:\n  in : {''.join(inp)!r}\n  out: {''.join(outs)!r}")
    build("test", "test", 4096, 1)
    build("heavy", "test", 4096, 2, lo=0.15, hi=0.3, clean_p=0.0)
    build("train", "train", 400000, 3)
