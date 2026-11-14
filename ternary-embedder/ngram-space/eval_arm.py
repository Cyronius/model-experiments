"""Evaluate a trained arm on the dev tasks: TS and agreement@10 at the 504 and
128 contracts, stored dims 1024 and 4096; optional mixed-index test with and
without calibration. Writes ngram-space/<stage>/arms/<arm>.json and per-query
rows (for paired tests) to training-v2/work/ngram-space/runs/<arm>/eval_rows.pkl.

Usage: python -I eval_arm.py ARM [--stage stage3] [--mixed] [--offset 0] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evalkit as E  # noqa: E402
from bmodel import Student  # noqa: E402
from train import S1, TIER, encode  # noqa: E402


def load_arm(arm, device):
    run = C.WORK / "runs" / arm
    a = json.load(open(run / "args.json"))
    k2 = np.load(S1 / "ranked_n2_keys.npy")[:TIER[a["n2"]]] if a.get("n2") else None
    k3 = np.load(S1 / "ranked_n3_keys.npy")[:TIER[a["n3"]]] if a.get("n3") else None
    m = Student(d=a["d"], n2_keys=k2, n3_keys=k3).to(device)
    if a.get("mlp"):
        m.add_mlp()
    if a.get("chunk"):
        m.add_chunking(a["chunk"])
    for _ in range(a.get("K", 0)):
        m.add_mixer_layer()
    m.load_state_dict(torch.load(run / "best.pt", map_location=device))
    m.eval()
    return m, a


def doc_fn_for(m, device, offset=0):
    cache = {}

    def fn(name, contract):
        k = (name, contract)
        if k not in cache:
            seqs = [s[:504] for s in C.encode(E.docs(name, contract))]
            seqs = [s if len(s) else np.array([C.UNK], np.uint16) for s in seqs]
            cache[k] = encode(m, seqs, device, offset=offset)
        return cache[k]
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--stage", default="stage3")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    m, args = load_arm(a.arm, a.device)
    fn = doc_fn_for(m, a.device, a.offset)
    summary, rows = E.full_eval(fn)
    res = {"arm": a.arm, "args": args, "offset": a.offset, "eval": summary}
    tj = C.WORK / "runs" / a.arm / "train.json"
    if tj.exists():
        t = json.load(open(tj))
        res["best_epoch"] = t["best_epoch"]
        res["val_loss"] = t["best_val_loss"]
        res["train_loss_sample"] = t.get("best_train_loss_9204_sample_no_dropout")
        res["history"] = t["history"]
    if a.mixed:
        import mixed as MX
        _, ptexts, _ = MX.round2_validation()
        pv = encode(m, [s[:504] if len(s) else np.array([C.UNK], np.uint16) for s in C.encode(ptexts)], a.device)
        cal = MX.fit_calibration(pv)
        res["calibration"] = cal
        res["mixed_raw"] = MX.mixed(fn)
        res["mixed_calibrated"] = MX.mixed(fn, cal)
    out = C.PLAN / a.stage / "arms"
    out.mkdir(parents=True, exist_ok=True)
    name = a.arm + (f"_{a.tag}" if a.tag else "")
    json.dump(res, open(out / f"{name}.json", "w"), indent=1)
    with open(C.WORK / "runs" / a.arm / f"eval_rows{('_' + a.tag) if a.tag else ''}.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    s = summary["c504_d1024"]
    print(f"{name}: core TS {s['core_ndcg10']:.2f} (TT {s['core_TT_ndcg10']:.2f}) agree {s['dev_agreement10']:.1f} "
          f"| c128 core {summary['c128_d1024']['core_ndcg10']:.2f} | d4096 core {summary['c504_d4096']['core_ndcg10']:.2f}"
          + (f" | val {res.get('val_loss'):.5f}" if res.get("val_loss") is not None else ""), flush=True)


if __name__ == "__main__":
    main()
