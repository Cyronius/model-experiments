"""Shared item `leakage-jaswanth` (claimed by Plan B, 2026-11-12) plus the
Jaswanth validation ids (contract section 2, `splits/`).

Rule (qwen-space/PLAN.md Stage 2, "Remove benchmark leakage"): drop any
training text that matches a document of the dev tasks (SciFact, NFCorpus,
ArguAna) or of the confirmation tasks in qwen-space/RESERVED.md (BEIR NQ,
Touche-2020; full corpora, not only the subsamples):
compare normalized text (shared.leak_key: lower case, collapsed whitespace,
first 200 characters) in both directions, and titles exactly.

Operational reading, fixed before looking at any match:
  * equal keys match;
  * "both directions": the training key is a prefix of a benchmark key, or a
    benchmark key is a prefix of the training key; prefix matches count only
    when the shorter key has at least 30 characters (so that a two-word
    training text is not removed for being the start of some document);
  * titles exactly: leak_key(training text) == leak_key(document title), for
    non-empty titles of at least 10 characters.
Benchmark document text = (title + " " + text).strip(), as the dev loader builds it.

Inputs: Plan B's converted Jaswanth texts (training-v2/work/ngram-space/jaswanth/texts.jsonl,
row ids in the shared manifest's row order), the dev data in
comparisons/qwen06b/data/, and the BEIR zips in training-v2/work/shared-qwen8b/reserved/
(read as data, never extracted or executed).

Outputs (training-v2/work/shared-qwen8b/):
  leakage/jaswanth-removed.tsv   row, sha256(text utf-8), benchmarks, kinds
  leakage/jaswanth-summary.json  counts per benchmark and kind
  splits/jaswanth-validation-ids.npy  row ids with shared.is_validation(text)
  splits/jaswanth-validation.json     count and sha256 of the id file
"""
import bisect
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

MINPREF = 30
MINTITLE = 10
SW = C.SHARED_WORK


def bench_docs():
    dev = C.ROOT / "comparisons/qwen06b/data"
    for name in ("scifact", "nfcorpus", "arguana"):
        with open(dev / name / "corpus.jsonl") as fh:
            yield name, (json.loads(line) for line in fh)
    for name, zp, inner in (("nq", SW / "reserved/nq.zip", "nq/corpus.jsonl"),
                            ("webis-touche2020", SW / "reserved/webis-touche2020.zip",
                             "webis-touche2020/corpus.jsonl")):
        z = zipfile.ZipFile(zp)
        yield name, (json.loads(line) for line in z.open(inner))


def main():
    sh = C.shared()
    t0 = time.time()
    texts = [json.loads(l) for l in open(C.WORK / "jaswanth" / "texts.jsonl")]
    N = len(texts)
    tkeys = [sh.leak_key(t) for t in texts]
    order = sorted(range(N), key=tkeys.__getitem__)
    T = [tkeys[i] for i in order]
    print(f"{time.time()-t0:.0f}s train keys {N}", flush=True)
    hits = {}          # row -> set of (bench, kind)
    summary = {}
    for name, docs in bench_docs():
        keys, titles, nd = [], set(), 0
        for d in docs:
            nd += 1
            title = (d.get("title") or "").strip()
            keys.append(sh.leak_key((title + " " + d.get("text", "")).strip()))
            if len(title) >= MINTITLE:
                titles.add(sh.leak_key(title))
        B = sorted(set(keys))
        cnt = {"exact_or_train_prefix": 0, "bench_prefix": 0, "title": 0}
        # training key equal to, or a prefix of, a benchmark key
        for i, k in enumerate(tkeys):
            if not k:
                continue
            j = bisect.bisect_left(B, k)
            if j < len(B) and B[j].startswith(k) and (B[j] == k or len(k) >= MINPREF):
                hits.setdefault(i, set()).add((name, "exact_or_train_prefix"))
                cnt["exact_or_train_prefix"] += 1
            if k in titles:
                hits.setdefault(i, set()).add((name, "title"))
                cnt["title"] += 1
        # benchmark key a proper prefix of training keys
        for b in B:
            if len(b) < MINPREF or len(b) >= 200:
                continue
            j = bisect.bisect_right(T, b)       # skip equal keys (counted above)
            while j < N and T[j].startswith(b):
                hits.setdefault(order[j], set()).add((name, "bench_prefix"))
                cnt["bench_prefix"] += 1
                j += 1
        rows_hit = sum(1 for v in hits.values() if any(x[0] == name for x in v))
        summary[name] = {"documents": nd, "rows_matched": rows_hit, **cnt}
        print(f"{time.time()-t0:.0f}s {name}: {summary[name]}", flush=True)
    (SW / "leakage").mkdir(exist_ok=True)
    with open(SW / "leakage" / "jaswanth-removed.tsv", "w") as fh:
        fh.write("row\tsha256\tbenchmarks\tkinds\n")
        for i in sorted(hits):
            v = sorted(hits[i])
            fh.write(f"{i}\t{hashlib.sha256(texts[i].encode('utf-8')).hexdigest()}\t"
                     f"{','.join(sorted({x[0] for x in v}))}\t{','.join(sorted({x[1] for x in v}))}\n")
    summ = {"source": "jaswanth", "rows": N, "removed_rows": len(hits), "per_benchmark": summary,
            "rule": __doc__.split("Inputs:")[0].strip(), "built_by": "Plan B",
            "date": time.strftime("%Y-%m-%d")}
    json.dump(summ, open(SW / "leakage" / "jaswanth-summary.json", "w"), indent=1)
    (SW / "splits").mkdir(exist_ok=True)
    val = np.asarray([i for i, t in enumerate(texts) if sh.is_validation(t)], np.int64)
    np.save(SW / "splits" / "jaswanth-validation-ids.npy", val)
    vh = hashlib.sha256(open(SW / "splits" / "jaswanth-validation-ids.npy", "rb").read()).hexdigest()
    json.dump({"source": "jaswanth", "rule": "int(sha256(text.encode('utf-8')).hexdigest()[:8], 16) % 128 == 0",
               "count": int(len(val)), "ids_file_sha256": vh, "row_order": "shared MANIFEST jaswanth.row_order",
               "built_by": "Plan B", "date": time.strftime("%Y-%m-%d")},
              open(SW / "splits" / "jaswanth-validation.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "rule"}, indent=1))
    print("validation ids", len(val), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
