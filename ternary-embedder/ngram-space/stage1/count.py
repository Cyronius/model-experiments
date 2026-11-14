"""Stage 1 step 2-3: count unigrams exactly, bigrams and trigrams with a
count-min sketch (2 rows x 2^27 uint32 per order = 1 GiB per order) and an
exact second pass over keys whose sketch count is >= 20; then select nested
tiers T1/T2/T3 per order.

Streams: directories holding ids.u16 + starts.npy (written by
tokenize_local.py and stage0/build_train.py). Optional per-stream token cap.

Usage: python -I stage1/count.py STREAM_DIR[:CAP_TOKENS] ...
"""
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.WORK / "stage1"
LOGW = 27
ROWS = 2
MULT = [np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F)]
THRESH = 20
BLOCK = 50_000_000
TIERS = {"T1": 250_000, "T2": 1_000_000, "T3": 2_000_000}


def blocks(streams):
    """Yield (flat ids, starts relative) blocks cut at document boundaries."""
    for d, cap in streams:
        ids = np.memmap(d / "ids.u16", dtype=np.uint16, mode="r")
        starts = np.load(d / "starts.npy")
        end_tok = len(ids) if cap is None else min(len(ids), cap)
        ndoc_end = np.searchsorted(starts, end_tok, side="right") - 1
        lo = 0
        while lo < ndoc_end:
            hi = np.searchsorted(starts, starts[lo] + BLOCK, side="right") - 1
            hi = max(hi, lo + 1)
            hi = min(hi, ndoc_end)
            s = starts[lo:hi + 1]
            yield np.asarray(ids[s[0]:s[-1]]), s - s[0]
            lo = hi


def hashes(k):
    return [((k * m) >> np.uint64(64 - LOGW)).astype(np.int64) for m in MULT]


def main():
    streams = []
    for a in sys.argv[1:]:
        p, _, cap = a.partition(":")
        streams.append((Path(p), int(cap) if cap else None))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    uni = np.zeros(C.VOCAB, np.int64)
    sk = {n: np.zeros((ROWS, 1 << LOGW), np.uint32) for n in (2, 3)}
    ntok = 0
    for flat, s in blocks(streams):
        ntok += len(flat)
        uni += np.bincount(flat, minlength=C.VOCAB)
        for n in (2, 3):
            k, _ = C.doc_keys(flat, s, n)
            for r, h in enumerate(hashes(k)):
                sk[n][r] += np.bincount(h, minlength=1 << LOGW).astype(np.uint32)
        print(f"pass1 {time.time()-t0:6.0f}s tokens {ntok:,}", flush=True)
    np.save(OUT / "unigram_counts.npy", uni)

    # pass 2: exact counts of sketch candidates
    acc = {n: (np.zeros(0, np.uint64), np.zeros(0, np.int64)) for n in (2, 3)}
    for flat, s in blocks(streams):
        for n in (2, 3):
            k, _ = C.doc_keys(flat, s, n)
            hs = hashes(k)
            est = np.minimum(sk[n][0][hs[0]], sk[n][1][hs[1]])
            k = k[est >= THRESH]
            u, c = np.unique(k, return_counts=True)
            ak, ac = acc[n]
            mk = np.concatenate([ak, u])
            mc = np.concatenate([ac, c])
            o = np.argsort(mk, kind="stable")
            mk, mc = mk[o], mc[o]
            uk, idx = np.unique(mk, return_index=True)
            acc[n] = (uk, np.add.reduceat(mc, idx))
        print(f"pass2 {time.time()-t0:6.0f}s cands {len(acc[2][0]):,} {len(acc[3][0]):,}", flush=True)
    del sk

    # selection
    tok = C.tokenizer()
    vocab = [tok.id_to_token(i) for i in range(C.VOCAB)]

    def is_punct(s):
        s = s[2:] if s.startswith("##") else s
        return len(s) > 0 and all(unicodedata.category(ch)[0] in "PS" for ch in s)

    top150 = set(np.argsort(-uni, kind="stable")[:150].tolist())
    boring = np.zeros(C.VOCAB, bool)
    for i in range(C.VOCAB):
        boring[i] = i in top150 or is_punct(vocab[i])
    meta = {"tokens": int(ntok), "streams": [f"{p}:{c}" for p, c in streams],
            "threshold": THRESH, "sketch": f"{ROWS}x2^{LOGW} uint32"}
    for n in (2, 3):
        k, c = acc[n]
        k, c = k[c >= THRESH], c[c >= THRESH]
        parts = np.stack([(k >> np.uint64(C.BITS * (n - 1 - j))) & np.uint64(C.MASK15)
                          for j in range(n)]).astype(np.int64)
        has_unk = (parts == C.UNK).any(0) | np.isin(parts, list(C.SPECIAL_IDS)).any(0)
        all_boring = boring[parts].all(0)
        keep = ~has_unk & ~all_boring
        meta[f"n{n}"] = {"exact_ge_thresh": int(len(k)), "drop_unk_special": int(has_unk.sum()),
                         "drop_boring": int((all_boring & ~has_unk).sum()), "kept": int(keep.sum())}
        k, c = k[keep], c[keep]
        o = np.lexsort((k, -c))          # by count desc, then key
        k, c = k[o], c[o]
        np.save(OUT / f"ranked_n{n}_keys.npy", k[:max(TIERS.values())])
        np.save(OUT / f"ranked_n{n}_counts.npy", c[:max(TIERS.values())])
        for t, sz in TIERS.items():
            meta[f"n{n}"][f"{t}_min_count"] = int(c[min(sz, len(c)) - 1]) if len(c) else 0
            meta[f"n{n}"][f"{t}_size"] = int(min(sz, len(c)))
        print(n, meta[f"n{n}"], flush=True)
    meta["top150"] = sorted(int(i) for i in top150)
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(OUT / "count_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
