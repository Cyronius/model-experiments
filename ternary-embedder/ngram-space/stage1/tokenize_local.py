"""Stage 1 step 1, local part: tokenize the local candidate corpora for counting.

Reads every `query` and `positive` string of
  training-v2/work/data-first/candidates/*.jsonl
  training-v2/work/broad-data/candidates/<source>/*.jsonl
  training-v2/work/broad-data/ready/train.jsonl
drops exact duplicate strings (across all sources), drops strings that fall in
the shared validation split (contract rule, applied to the string), cuts each
to its first 504 content tokens, and writes a flat uint16 id stream plus
document starts and a source id per document to the work folder.

Usage: python -I stage1/tokenize_local.py   (from anywhere)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.WORK / "stage1" / "local"
MAXTOK = 504
BATCH = 20000


def is_validation(text: str) -> bool:
    # contract section 2 rule (identical to shared.is_validation; checked later)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 128 == 0


def sources():
    w = C.ROOT / "training-v2" / "work"
    out = []
    for f in sorted((w / "data-first" / "candidates").glob("*.jsonl")):
        out.append(("df/" + f.stem, [f]))
    for d in sorted((w / "broad-data" / "candidates").iterdir()):
        fs = sorted(d.glob("*.jsonl"))
        if fs:
            out.append(("bd/" + d.name, fs))
    out.append(("ready/train", [w / "broad-data" / "ready" / "train.jsonl"]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    ids_f = open(OUT / "ids.u16", "wb")
    starts = [0]
    src_of = []
    stats = {}
    total = 0
    t0 = time.time()
    srcs = sources()
    buf, bufsrc = [], []

    def flush():
        nonlocal total
        if not buf:
            return
        enc = C.encode(buf)
        for e, s in zip(enc, bufsrc):
            e = e[:MAXTOK]
            ids_f.write(e.tobytes())
            total += len(e)
            starts.append(total)
            src_of.append(s)
            stats[srcs[s][0]]["tokens"] += len(e)
        buf.clear()
        bufsrc.clear()

    for si, (name, files) in enumerate(srcs):
        st = stats[name] = {"files": len(files), "strings": 0, "dup": 0, "validation": 0,
                            "kept": 0, "tokens": 0}
        for f in files:
            with open(f, "rb") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    for fld in ("query", "positive"):
                        t = row.get(fld)
                        if not isinstance(t, str) or not t.strip():
                            continue
                        st["strings"] += 1
                        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
                        if h in seen:
                            st["dup"] += 1
                            continue
                        seen.add(h)
                        if is_validation(t):
                            st["validation"] += 1
                            continue
                        st["kept"] += 1
                        buf.append(t)
                        bufsrc.append(si)
                        if len(buf) >= BATCH:
                            flush()
        flush()
        print(f"{time.time()-t0:7.0f}s {name}: {st}  total tokens {total:,}", flush=True)
    ids_f.close()
    np.save(OUT / "starts.npy", np.asarray(starts, dtype=np.int64))
    np.save(OUT / "source.npy", np.asarray(src_of, dtype=np.uint8))
    json.dump({"sources": [s[0] for s in srcs], "stats": stats, "tokens": total,
               "docs": len(src_of), "max_tokens": MAXTOK,
               "seconds": time.time() - t0}, open(OUT / "meta.json", "w"), indent=1)
    print("done", total, len(src_of))


if __name__ == "__main__":
    main()
