"""Plan B evaluation on the dev tasks (SciFact, NFCorpus core; ArguAna secondary).

Uses `training-v2/round2_metrics.py::evaluate` and
`retrieval-diagnostics/diagnose.py::paired` unchanged for nDCG@10/recall@10.
Agreement@10 (PLAN section 5) is computed here from the per-query top-10 lists
that `evaluate` returns: share of the TT top-10 also in the TS top-10, both
ranked over the full corpus.

Teacher vectors come from `shared.embed(plan="B")` (shared cache): documents
as bare strings cut to the contract, queries as
`Instruct: {instruction}\nQuery:{query}` (query protocol).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TASKS = ("SciFact", "NFCorpus", "ArguAna")
CORE = ("SciFact", "NFCorpus")
CONTRACT_TOKENS = {504: 504, 128: 126}
DEVDIR = C.WORK / "dev"


def _metrics():
    for p in ("training-v2", "retrieval-diagnostics", "tern-embed"):
        q = str(C.ROOT / p)
        if q not in sys.path:
            sys.path.insert(0, q)
    import round2_metrics
    import diagnose
    return round2_metrics.evaluate, diagnose.paired


@lru_cache(None)
def task(name):
    return C.shared().dev_task(name)


@lru_cache(None)
def docs(name, contract):
    """Document strings under the contract (504 or 128)."""
    f = DEVDIR / f"{name}-{contract}-texts.json"
    if f.exists():
        return json.load(open(f))
    sh = C.shared()
    out = [sh.cut_to_contract(t, CONTRACT_TOKENS[contract]) for t in task(name)["texts"]]
    DEVDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f, "w"))
    return out


def teacher_docs(name, contract):
    f = DEVDIR / f"{name}-{contract}-tdocs.npy"
    if f.exists():
        return np.load(f)
    v = C.embed(docs(name, contract))
    np.save(f, v)
    return v


def teacher_queries(name):
    f = DEVDIR / f"{name}-tqueries.npy"
    if f.exists():
        return np.load(f)
    sh = C.shared()
    d = task(name)
    v = C.embed([sh.query_input(q, d["instruction"]) for q in d["queries"]])
    np.save(f, v)
    return v


def cut(x, k):
    return C.unit(np.asarray(x, np.float32)[:, :k])


def score(name, doc_vecs, dim):
    """evaluate() metrics and rows for teacher queries against doc_vecs at stored dim."""
    evaluate, _ = _metrics()
    q = cut(teacher_queries(name), dim)
    d = cut(doc_vecs, dim)
    return evaluate(q @ d.T, task(name))


_tt_cache = {}


def tt_rows(name, contract, dim):
    k = (name, contract, dim)
    if k not in _tt_cache:
        _tt_cache[k] = score(name, teacher_docs(name, contract), dim)
    return _tt_cache[k]


def agreement_rows(rows_tt, rows_ts):
    return np.array([len(set(a["top10"]) & set(b["top10"])) / 10.0 for a, b in zip(rows_tt, rows_ts)])


def full_eval(doc_fn, contracts=(504, 128), dims=(1024, 4096)):
    """doc_fn(name, contract) -> [n_docs, 4096] student vectors.
    Returns (summary dict, per-query rows dict for paired comparisons)."""
    out, rows = {}, {}
    for contract in contracts:
        for dim in dims:
            key = f"c{contract}_d{dim}"
            res, agr_all = {}, []
            for name in TASKS:
                m, r = score(name, doc_fn(name, contract), dim)
                mt, rt = tt_rows(name, contract, dim)
                ag = agreement_rows(rt, r)
                agr_all.append(ag)
                res[name] = {"ndcg10": m["ndcg10"], "recall10": m["recall10"],
                             "TT_ndcg10": mt["ndcg10"], "agreement10": 100 * float(ag.mean())}
                rows[(key, name)] = r
            res["core_ndcg10"] = float(np.mean([res[t]["ndcg10"] for t in CORE]))
            res["core_TT_ndcg10"] = float(np.mean([res[t]["TT_ndcg10"] for t in CORE]))
            res["dev_agreement10"] = 100 * float(np.concatenate(agr_all).mean())
            rows[(key, "agreement")] = np.concatenate(agr_all)
            out[key] = res
    return out, rows


def paired_agreement(left, right):
    """Paired bootstrap of dev agreement (right - left), same scheme as diagnose.paired."""
    delta = np.asarray(right) - np.asarray(left)
    rng = np.random.default_rng(0)
    s = delta[rng.integers(0, len(delta), size=(4000, len(delta)))].mean(1) * 100
    return {"agreement_delta_right_minus_left": float(delta.mean() * 100),
            "paired_bootstrap_95_ci": np.percentile(s, [2.5, 97.5]).tolist()}


def paired_core(rows_left, rows_right, key="c504_d1024"):
    """Paired bootstrap of core nDCG (mean over SciFact and NFCorpus deltas)."""
    _, paired = _metrics()
    return {t: paired(rows_left[(key, t)], rows_right[(key, t)]) for t in TASKS}
