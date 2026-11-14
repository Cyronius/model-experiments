"""Stage 1 step 4: most frequent original surface string of each selected
bigram and trigram (T3), found with the tokenizer's character offsets.

Surface of an n-gram ending at position i = text[offset[i-n+1].start : offset[i].end]
(original casing and spacing). Counted over the Jaswanth training texts plus
the data-first local candidates (a sample of the counting corpus; exact
counting of every (key, surface) pair over the full corpus is not needed to
pick the mode). N-grams never seen get the tokenizer's decode as surface and
are counted in the report.

Usage: python -I stage1/surfaces.py
"""
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

S1 = C.WORK / "stage1"
KEYS = {}


def init():
    for n in (2, 3):
        KEYS[n] = np.sort(np.load(S1 / f"ranked_n{n}_keys.npy"))
    C.tokenizer()


def work(texts):
    out = {2: defaultdict(Counter), 3: defaultdict(Counter)}
    ids, offs = C.encode(texts, offsets=True)
    for t, e, o in zip(texts, ids, offs):
        e = e[:504]
        for n in (2, 3):
            if len(e) < n:
                continue
            k = C.bigram_keys(e) if n == 2 else C.trigram_keys(e)
            ks = KEYS[n]
            j = np.searchsorted(ks, k)
            j[j >= len(ks)] = 0
            hit = np.nonzero(ks[j] == k)[0]
            for p in hit.tolist():
                out[n][int(k[p])][t[o[p][0]:o[p + n - 1][1]]] += 1
    return out


def text_batches():
    jw = C.WORK / "jaswanth"
    train = np.load(C.WORK / "stage0" / "train_rows.npy")
    keep = np.zeros(len(np.load(jw / "ntok.npy")), bool)
    keep[train] = True
    batch = []
    with open(jw / "texts.jsonl") as fh:
        for i, line in enumerate(fh):
            if keep[i]:
                batch.append(json.loads(line))
                if len(batch) == 2000:
                    yield batch
                    batch = []
    for f in sorted((C.ROOT / "training-v2/work/data-first/candidates").glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                for fld in ("query", "positive"):
                    t = r.get(fld)
                    if isinstance(t, str) and t.strip():
                        batch.append(t)
                if len(batch) >= 2000:
                    yield batch
                    batch = []
    if batch:
        yield batch


NW = 24


def shard(w):
    """Worker w reads all texts but records only keys with key % NW == w,
    so no merging is needed. Returns {n: {key: most frequent surface}}."""
    init()
    for n in (2, 3):
        KEYS[n] = KEYS[n][KEYS[n] % np.uint64(NW) == np.uint64(w)]
    acc = {2: defaultdict(Counter), 3: defaultdict(Counter)}
    for b, texts in enumerate(text_batches()):
        res = work(texts)
        for n in (2, 3):
            for k, c in res[n].items():
                acc[n][k].update(c)
        if w == 0 and b % 100 == 0:
            print(f"worker0 batch {b}", flush=True)
    return {n: {k: c.most_common(1)[0][0] for k, c in acc[n].items()} for n in (2, 3)}


def main():
    import os
    os.environ["RAYON_NUM_THREADS"] = "1"
    t0 = time.time()
    acc = {2: {}, 3: {}}
    with Pool(NW) as pool:
        for res in pool.imap_unordered(shard, range(NW)):
            for n in (2, 3):
                acc[n].update(res[n])
            print(f"{time.time()-t0:6.0f}s shard done; keys {len(acc[2]):,} {len(acc[3]):,}", flush=True)
    tok = C.tokenizer()
    meta = {}
    for n in (2, 3):
        ranked = np.load(S1 / f"ranked_n{n}_keys.npy")
        surf, seen = [], 0
        for k in ranked.tolist():
            c = acc[n].get(int(k))
            if c is not None:
                surf.append(c)
                seen += 1
            else:
                surf.append(tok.decode(list(C.unpack(k, n))))
        with open(S1 / f"ranked_n{n}_surface.jsonl", "w") as fh:
            for s in surf:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        tiers = {"T1": 250_000, "T2": 1_000_000, "T3": 2_000_000}
        meta[f"n{n}"] = {"seen": seen, "unseen_decode_fallback": len(ranked) - seen,
                         **{f"{t}_unseen": int(sz - sum(1 for k in ranked[:sz].tolist() if int(k) in acc[n]))
                            for t, sz in tiers.items()}}
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(S1 / "surface_meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
