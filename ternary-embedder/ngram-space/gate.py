"""The B1/B2 test between two evaluated arms (fixed in PLAN section 6, Stage 3):
the larger arm passes against the smaller if dev agreement@10 (504 contract,
stored dim 1024) rises by at least 1.0 point with the paired 95% bootstrap
interval excluding zero, and its core TS (mean nDCG@10 of SciFact and
NFCorpus) is not more than 0.3 lower.

Usage: python -I gate.py SMALL_ARM LARGE_ARM [--key c504_d1024]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evalkit as E  # noqa: E402


def rows(arm, tag=""):
    with open(C.WORK / "runs" / arm / f"eval_rows{('_' + tag) if tag else ''}.pkl", "rb") as fh:
        return pickle.load(fh)


def test(small, large, key="c504_d1024", stag="", ltag=""):
    a, b = rows(small, stag), rows(large, ltag)
    agr = E.paired_agreement(a[(key, "agreement")], b[(key, "agreement")])
    core_small = np.mean([np.mean([r["ndcg10"] for r in a[(key, t)]]) for t in E.CORE]) * 100
    core_large = np.mean([np.mean([r["ndcg10"] for r in b[(key, t)]]) for t in E.CORE]) * 100
    nd = E.paired_core(a, b, key)
    lo, hi = agr["paired_bootstrap_95_ci"]
    ok_agr = agr["agreement_delta_right_minus_left"] >= 1.0 and lo > 0
    ok_ts = core_large >= core_small - 0.3
    return {"small": small, "large": large, "key": key,
            "agreement": agr, "core_TS_small": core_small, "core_TS_large": core_large,
            "core_TS_delta": core_large - core_small, "ndcg_paired": nd,
            "agreement_rule_pass": bool(ok_agr), "ts_rule_pass": bool(ok_ts), "pass": bool(ok_agr and ok_ts)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("small")
    ap.add_argument("large")
    ap.add_argument("--key", default="c504_d1024")
    a = ap.parse_args()
    r = test(a.small, a.large, a.key)
    print(json.dumps({k: v for k, v in r.items() if k != "ndcg_paired"}, indent=1))
