"""Stage 2 step 3: linear ceiling. Each dev document's teacher vector t is
replaced by normalize(V_d V_d^T (t - mu) + mu) for d = 512 and 1024; report TS
(nDCG@10) and dev agreement@10 at both contracts and stored dims 1024 / 4096.
Also records TT (the teacher itself) through the same code.

Usage: DEEPINFRA_API_KEY=... python -I stage2/ceiling.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import evalkit as E  # noqa: E402

S2 = C.WORK / "stage2"


def main():
    mu = np.load(S2 / "mu.npy")
    V = np.load(S2 / "V.npy")
    sh = C.shared()
    before = sh.spend("B")
    res = {}
    for name, fn in [("TT", lambda t, c: E.teacher_docs(t, c).astype(np.float32))] + [
        (f"linear_d{d}", (lambda d: lambda t, c: C.unit(
            (E.teacher_docs(t, c).astype(np.float32) - mu) @ V[:, :d] @ V[:, :d].T + mu))(d))
            for d in (512, 1024)]:
        out, _ = E.full_eval(fn)
        res[name] = out
        for k, v in out.items():
            print(name, k, f"core {v['core_ndcg10']:.2f} (TT {v['core_TT_ndcg10']:.2f})",
                  " ".join(f"{t} {v[t]['ndcg10']:.2f}" for t in E.TASKS), f"agree {v['dev_agreement10']:.1f}",
                  flush=True)
    after = sh.spend("B")
    res["spend_B_before"], res["spend_B_after"] = before, after
    json.dump(res, open(C.PLAN / "stage2" / "ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
