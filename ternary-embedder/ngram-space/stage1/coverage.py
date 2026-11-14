"""Stage 1 step 5: coverage of the shared validation texts (Jaswanth validation
split) per tier and order, for texts of <=128 tokens and longer; unigram
frequencies p[t]; table bytes per tier.

Coverage denominator: positions that end an n-gram of that order inside the
document (position >= n-1), documents cut at 504 tokens.

Usage: python -I stage1/coverage.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

S1 = C.WORK / "stage1"
TIERS = {"T1": 250_000, "T2": 1_000_000, "T3": 2_000_000}


def main():
    jw = C.WORK / "jaswanth"
    ids = np.memmap(jw / "ids.u16", dtype=np.uint16, mode="r")
    starts = np.load(jw / "starts.npy")
    val = np.load(C.WORK / "stage0" / "val_rows.npy")
    ntok = np.load(jw / "ntok.npy")
    res = {"validation_texts": int(len(val))}
    groups = {"le128": val[ntok[val] <= 128], "gt128": val[ntok[val] > 128], "all": val}
    for g, rows in groups.items():
        flat = np.concatenate([ids[starts[r]:starts[r + 1]] for r in rows])
        lens = np.asarray([starts[r + 1] - starts[r] for r in rows])
        st = np.concatenate([[0], np.cumsum(lens)])
        res[g] = {"texts": int(len(rows)), "tokens": int(len(flat))}
        res[g]["unk_share"] = float((flat == C.UNK).mean())
        for n in (2, 3):
            k, _ = C.doc_keys(flat, st, n)
            ranked = np.load(S1 / f"ranked_n{n}_keys.npy")
            for t, sz in TIERS.items():
                ks = np.sort(ranked[:sz])
                j = np.searchsorted(ks, k)
                j[j >= len(ks)] = 0
                res[g][f"n{n}_{t}"] = float((ks[j] == k).mean())
            res[g][f"n{n}_positions"] = int(len(k))
    uni = np.load(S1 / "unigram_counts.npy").astype(np.float64)
    p = uni / uni.sum()
    np.save(S1 / "unigram_p.npy", p)
    res["unigram"] = {"tokens": int(uni.sum()), "ids_seen": int((uni > 0).sum()),
                      "p_max": float(p.max()), "argmax": int(p.argmax())}
    res["table_bytes_w512"] = {}
    for t, sz in TIERS.items():
        rows = 2 * sz            # bigram + trigram tiers of equal size
        res["table_bytes_w512"][t] = {"rows_bi_plus_tri": rows,
                                      "float32_GB": rows * 512 * 4 / 1e9,
                                      "int8_plus_scale_GB": rows * (512 + 4) / 1e9,
                                      "keys_GB_at_load_0.5": 2 * rows * 8 / 1e9}
    res["unigram_table_bytes"] = {"d1024_float32_MB": C.VOCAB * 1024 * 4 / 1e6,
                                  "d1024_int8_MB": C.VOCAB * (1024 + 4) / 1e6}
    json.dump(res, open(S1 / "coverage.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
