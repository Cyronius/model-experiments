"""Stage 0 step 4: Plan B training set from the Jaswanth set.

Keep rows that fit the 504 contract in full (<= 504 WordPiece content tokens,
so cut_to_contract leaves the text unchanged and teacher and student saw the
same text); drop the shared leakage list and the shared validation split.
Validation rows used by Plan B = shared validation ids that also fit 504.

Writes training-v2/work/ngram-space/stage0/{train_rows,val_rows}.npy and
ngram-space/stage0/result.json (+ RESULTS.md written by hand from it).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

SW = C.SHARED_WORK


def main():
    jw = C.WORK / "jaswanth"
    ntok = np.load(jw / "ntok.npy")
    valid_mine = np.load(jw / "valid.npy")
    N = len(ntok)
    val_ids = np.load(SW / "splits" / "jaswanth-validation-ids.npy")
    valid = np.zeros(N, bool)
    valid[val_ids] = True
    assert (valid == valid_mine).all(), "validation split mismatch"
    leak = np.zeros(N, bool)
    with open(SW / "leakage" / "jaswanth-removed.tsv") as fh:
        next(fh)
        for line in fh:
            leak[int(line.split("\t", 1)[0])] = True
    fit = (ntok <= 504) & (ntok > 0)          # empty texts (no tokens) cannot be embedded
    train = fit & ~valid & ~leak
    val = fit & valid
    out = C.WORK / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_rows.npy", np.nonzero(train)[0])
    np.save(out / "val_rows.npy", np.nonzero(val)[0])
    q = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    res = {
        "rows": int(N),
        "fit_504": int(fit.sum()), "dropped_over_504": int((ntok > 504).sum()), "dropped_empty": int((ntok == 0).sum()),
        "validation_split": int(valid.sum()), "validation_fit_504": int(val.sum()),
        "leakage_rows": int(leak.sum()), "leakage_and_fit_and_not_validation": int((leak & fit & ~valid).sum()),
        "train_rows": int(train.sum()),
        "train_tokens": int(ntok[train].sum()),
        "train_token_length_percentiles": dict(zip(map(str, q), np.percentile(ntok[train], q).astype(int).tolist())),
        "train_share_le_128": float((ntok[train] <= 128).mean()),
        "train_share_le_126": float((ntok[train] <= 126).mean()),
        "all_rows_token_length_percentiles": dict(zip(map(str, q), np.percentile(ntok, q).astype(int).tolist())),
        "histogram_edges": [0, 16, 32, 64, 128, 256, 504],
        "train_histogram": np.histogram(ntok[train], [0, 16, 32, 64, 128, 256, 505])[0].tolist(),
    }
    json.dump(res, open(C.PLAN / "stage0" / "result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
