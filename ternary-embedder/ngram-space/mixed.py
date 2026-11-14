"""Mixed-index test and calibration (Plan A Stage 2, implemented in Plan B's folder).

Mixed index: a random share p of a dev task's documents is provisional
(student vector), the rest refined (teacher vector); all are ranked against
teacher queries at stored dimension 1024, 504 contract. p in {1, 10, 50}%,
seeds 0-4. Reported per p (mean over seeds, core = SciFact + NFCorpus):
  * overall nDCG@10 (round2_metrics.evaluate);
  * recall ratio: relevant provisional documents found in the top 10, divided
    by how many of the same documents are in the top 10 when all documents are
    refined (pooled over queries and seeds);
  * share of top-10 slots held by provisional documents.
Calibration: one affine map a*s + b on provisional scores, fit by least
squares to the teacher's score of the same (query, passage) pair, on the
round-2 teacher-cache validation split (queries with shared.is_validation
true, each against all validation passages; passages under 127 Qwen tokens;
1024 prefix). The training-source validation split, never the dev tasks.
Gate G2 (Plan A, = B4) at p = 10%: recall ratio >= 0.85 and provisional
top-10 share <= 15%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evalkit as E  # noqa: E402

PS = (0.01, 0.10, 0.50)
SEEDS = range(5)
DIM = 1024


def round2_validation():
    """(query vectors [nq,4096] f32, passage texts, passage teacher vectors [np,4096] f32)."""
    f = C.WORK / "calib" / "round2-val.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return z["q"], json.load(open(f.with_suffix(".json"))), z["p"]
    import hashlib
    sh = C.shared()
    R2 = C.ROOT / "training-v2/work/round2"
    man = json.loads((R2 / "teacher-cache/manifest.json").read_text())
    data = R2 / "balanced-refined.jsonl"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == man["prompted_metadata"]["data_sha256"]
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    from tokenizers import Tokenizer
    qtok = Tokenizer.from_file(str(next((C.ROOT / "cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots")
                                        .iterdir()) / "tokenizer.json"))
    P = np.load(R2 / "teacher-cache/passages.npy", mmap_mode="r")
    Q = np.load(R2 / "teacher-cache/prompted-queries.npy", mmap_mode="r")
    qi = [i for i, r in enumerate(rows) if sh.is_validation(r["query"])]
    pidx, ptexts, seen = [], [], set()
    for i in qi:
        for col, fld in enumerate(("positive", "negative")):
            t = rows[i].get(fld)
            if not t or t in seen or len(qtok.encode(t).ids) >= 127:
                continue
            seen.add(t)
            pidx.append((i, col))
            ptexts.append(t)
    q = np.asarray(Q[qi], np.float32)
    p = np.stack([np.asarray(P[i, c], np.float32) for i, c in pidx])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, q=q, p=p)
    json.dump(ptexts, open(f.with_suffix(".json"), "w"))
    return q, ptexts, p


def fit_calibration(student_vecs):
    """student_vecs: [np, 4096] unit vectors for round2_validation() passages."""
    q, _, p = round2_validation()
    qd, pd, sd = E.cut(q, DIM), E.cut(p, DIM), E.cut(student_vecs, DIM)
    st = (qd @ sd.T).ravel()
    tt = (qd @ pd.T).ravel()
    A = np.stack([st, np.ones_like(st)], 1)
    (a, b), *_ = np.linalg.lstsq(A.astype(np.float64), tt.astype(np.float64), rcond=None)
    return {"a": float(a), "b": float(b), "pairs": int(len(st)), "queries": int(len(q)),
            "passages": int(len(p)), "rmse_before": float(np.sqrt(((st - tt) ** 2).mean())),
            "rmse_after": float(np.sqrt(((a * st + b - tt) ** 2).mean()))}


def mixed(student_doc_fn, calib=None, contract=504):
    """student_doc_fn(name, contract) -> [n, 4096] student vectors."""
    evaluate, _ = E._metrics()
    out = {}
    for name in E.TASKS:
        d = E.task(name)
        qv = E.cut(E.teacher_queries(name), DIM)
        T = qv @ E.cut(E.teacher_docs(name, contract), DIM).T
        S = qv @ E.cut(student_doc_fn(name, contract), DIM).T
        if calib is not None:
            S = calib["a"] * S + calib["b"]
        cid = {c: i for i, c in enumerate(d["cids"])}
        rel = [[cid[c] for c, g in d["qrels"][q].items() if g > 0 and c in cid and c != q] for q in d["qids"]]
        # self-document exclusion (ArguAna) as in evaluate()
        selfdoc = [cid.get(q, -1) for q in d["qids"]]
        Tm = T.copy()
        for i, s in enumerate(selfdoc):
            if s >= 0:
                Tm[i, s] = -np.inf
        top_ref = np.argsort(-Tm, axis=1)[:, :10]
        res = {}
        for p in PS:
            nd, found_p, found_r, slots, slots_p = [], 0, 0, 0, 0
            for seed in SEEDS:
                prov = np.random.default_rng(seed).random(T.shape[1]) < p
                M = np.where(prov[None, :], S, T)
                m, _ = evaluate(M, d)
                nd.append(m["ndcg10"])
                Mm = M.copy()
                for i, s in enumerate(selfdoc):
                    if s >= 0:
                        Mm[i, s] = -np.inf
                top = np.argsort(-Mm, axis=1)[:, :10]
                for i in range(len(rel)):
                    rp = [r for r in rel[i] if prov[r]]
                    if rp:
                        found_p += len(set(rp) & set(top[i].tolist()))
                        found_r += len(set(rp) & set(top_ref[i].tolist()))
                slots += top.size
                slots_p += int(prov[top].sum())
            res[f"p{int(p*100)}"] = {"ndcg10": float(np.mean(nd)), "ndcg10_seeds": nd,
                                     "relevant_provisional_in_top10": found_p,
                                     "same_docs_in_top10_when_refined": found_r,
                                     "recall_ratio": found_p / found_r if found_r else None,
                                     "provisional_top10_share": slots_p / slots}
        res["all_refined_ndcg10"] = evaluate(T, d)[0]["ndcg10"]
        out[name] = res
    for p in PS:
        k = f"p{int(p*100)}"
        fp = sum(out[t][k]["relevant_provisional_in_top10"] for t in E.CORE)
        fr = sum(out[t][k]["same_docs_in_top10_when_refined"] for t in E.CORE)
        out.setdefault("core", {})[k] = {
            "ndcg10": float(np.mean([out[t][k]["ndcg10"] for t in E.CORE])),
            "recall_ratio": fp / fr if fr else None,
            "provisional_top10_share": float(np.mean([out[t][k]["provisional_top10_share"] for t in E.CORE]))}
    c = out["core"]["p10"]
    out["G2_at_p10"] = {"recall_ratio_ge_0.85": bool(c["recall_ratio"] is not None and c["recall_ratio"] >= 0.85),
                        "top10_share_le_0.15": bool(c["provisional_top10_share"] <= 0.15)}
    out["G2_at_p10"]["pass"] = all(out["G2_at_p10"].values())
    return out
