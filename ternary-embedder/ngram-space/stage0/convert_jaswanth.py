"""Stage 0 step 4, heavy part: convert the shared Jaswanth download (read only,
as data) into Plan B's work-folder arrays.

Row id = running index over the parquet files sorted by (part, shard) parsed
from `train_part_{part}-{shard}-of-00002.parquet` (part like "2_3" -> (2, 3)),
rows in file order: the order recorded in the shared MANIFEST.json.

Writes to training-v2/work/ngram-space/jaswanth/:
  codes.npy     [N, 4096] int8 codes as stored
  absmax.f32    [N] abs_max_val (float32)
  texts.jsonl   one JSON string per row (the text exactly as stored)
  ntok.npy      [N] WordPiece content-token count (no special tokens)
  valid.npy     [N] bool, contract validation split (shared.is_validation)
  ids.u16 + starts.npy   token ids of every row, cut at 504 tokens
  meta.json

Usage: python -I stage0/convert_jaswanth.py
"""
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

SRC = C.SHARED_WORK / "jaswanth" / "data"
OUT = C.WORK / "jaswanth"
D = 4096


def file_key(p):
    m = re.search(r"train_part_([\d_]+)-(\d+)-of", p.name)
    return tuple(int(x) for x in m[1].split("_")), int(m[2])


def load(path):
    t = pq.read_table(path, columns=["text", "embedding", "abs_max_val"])
    texts = t.column("text").to_pylist()
    emb = t.column("embedding").combine_chunks()
    vals = emb.values.to_numpy(zero_copy_only=False)
    assert len(vals) == len(texts) * D, (path, len(vals))
    assert vals.min() >= -128 and vals.max() <= 127
    codes = vals.astype(np.int8).reshape(len(texts), D)
    am = np.asarray(t.column("abs_max_val").to_numpy(), np.float32)
    return str(path), texts, codes, am


def main():
    sh = C.shared()
    files = sorted(SRC.glob("*.parquet"), key=file_key)
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    N = sum(counts)
    OUT.mkdir(parents=True, exist_ok=True)
    codes = np.lib.format.open_memmap(OUT / "codes.npy", mode="w+", dtype=np.int8, shape=(N, D))
    absmax = np.zeros(N, np.float32)
    ntok = np.zeros(N, np.int32)
    valid = np.zeros(N, bool)
    off = dict(zip(map(str, files), np.cumsum([0] + counts[:-1])))
    t0 = time.time()
    texts_all = [None] * N
    with Pool(12) as pool:
        for i, (p, texts, c, am) in enumerate(pool.imap_unordered(load, files)):
            o = int(off[p])
            codes[o:o + len(texts)] = c
            absmax[o:o + len(texts)] = am
            texts_all[o:o + len(texts)] = texts
            if i % 10 == 0:
                print(f"{time.time()-t0:6.0f}s read {i+1}/{len(files)}", flush=True)
    codes.flush()
    with open(OUT / "texts.jsonl", "w") as fh:
        for t in texts_all:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    ids_f = open(OUT / "ids.u16", "wb")
    starts = [0]
    tot = 0
    for b in range(0, N, 20000):
        part = texts_all[b:b + 20000]
        for j, (t, e) in enumerate(zip(part, C.encode(part))):
            ntok[b + j] = len(e)
            valid[b + j] = sh.is_validation(t)
            e = e[:504]
            ids_f.write(e.tobytes())
            tot += len(e)
            starts.append(tot)
    ids_f.close()
    np.save(OUT / "absmax.npy", absmax)
    np.save(OUT / "ntok.npy", ntok)
    np.save(OUT / "valid.npy", valid)
    np.save(OUT / "starts.npy", np.asarray(starts, np.int64))
    meta = {"rows": N, "files": [f.name for f in files], "rows_per_file": counts,
            "tokens_cut504": tot, "validation_rows": int(valid.sum()),
            "fit_504": int((ntok <= 504).sum()), "seconds": time.time() - t0,
            "revision": (C.SHARED_WORK / "jaswanth.revision").read_text().strip()}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
