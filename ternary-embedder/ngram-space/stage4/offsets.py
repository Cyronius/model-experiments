"""Stage 4 step 3: boundary robustness. Embed the dev documents (504 contract)
with chunk-grid offsets 0, 5 and 11 and report the cosine between the offset
variants of the same document (mean and 5th percentile, per pair of offsets,
and documents with at least one full chunk boundary only, for information).
The TS spread comes from eval_arm.py --offset N --tag offN.

Usage: python -I stage4/offsets.py ARM
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import evalkit as E  # noqa: E402
from eval_arm import load_arm  # noqa: E402
from train import encode  # noqa: E402

OFFS = (0, 5, 11)


def main(arm):
    m, a = load_arm(arm, "cpu")
    seqs, per_task = [], {}
    for name in E.TASKS:
        s = [x[:504] for x in C.encode(E.docs(name, 504))]
        s = [x if len(x) else np.array([C.UNK], np.uint16) for x in s]
        per_task[name] = (len(seqs), len(seqs) + len(s))
        seqs += s
    lens = np.array([len(x) for x in seqs])
    V = {o: encode(m, seqs, "cpu", offset=o) for o in OFFS}
    res = {"arm": arm, "chunk": a.get("chunk"), "n_docs": len(seqs), "pairs": {}}
    long_ = lens > a.get("chunk", 0)
    for i, o1 in enumerate(OFFS):
        for o2 in OFFS[i + 1:]:
            c = (V[o1] * V[o2]).sum(1)
            res["pairs"][f"{o1}-{o2}"] = {
                "mean": float(c.mean()), "p5": float(np.percentile(c, 5)), "min": float(c.min()),
                "mean_docs_longer_than_chunk": float(c[long_].mean()),
                "p5_docs_longer_than_chunk": float(np.percentile(c[long_], 5))}
    allc = np.concatenate([(V[x] * V[y]).sum(1) for x, y in ((0, 5), (0, 11), (5, 11))])
    res["all_pairs"] = {"mean": float(allc.mean()), "p5": float(np.percentile(allc, 5))}
    json.dump(res, open(C.PLAN / "stage4" / f"offsets_{arm}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
