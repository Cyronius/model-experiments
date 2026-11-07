"""Merge the accepted rewrites of the v5 production runs into one candidate set.

Reads accepted.jsonl of each --runs directory in data/out/doc-teacher-v5/ and writes
<out>/accepted.jsonl and <out>/summary.json. Each record keeps its fields and gets the
second meaning checks as review flags: review_flags.second_judge (second_judge_or.jsonl,
an OpenRouter judge of another family) and review_flags.local_judge (second_judge.jsonl,
the local model, when it checked that rewrite). None means not checked. Nothing is
removed for a flag; a passage id may appear once only, and the Rust Book never.
--repairs names repair runs (build_repair_v5.py, then pipeline_v5.py with --judge2): an
accepted repair whose text differs from the flagged rewrite replaces it, keeping the old
one in repaired_from. A repair that returns the same text does not clear the flag, and a
repair that puts a literal inside another word than in the source (a marker swap that the
judges cannot see, because the judge message numbers the rewrite's markers afresh) is not
applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from synthesize import mask_literals  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "data/out/doc-teacher-v5"


def glued(text: str) -> dict[str, set]:
    """For each literal, the word characters it is joined to on each side: 'utf-' and '-be' in utf-16-be."""
    marked, values = mask_literals(text)
    out = defaultdict(set)
    for m in re.finditer(r"\[\[L(\d+)\]\]", marked):
        before = re.search(r"[\w\-/]*$", marked[:m.start()])[0]
        after = re.match(r"[\w\-/]*", marked[m.end():])[0]
        out[values[int(m[1])]].add((re.sub(r"\[\[L\d+\]\]", "#", before), re.sub(r"\[\[L\d+\]\]", "#", after)))
    return out


def literal_moved(source: str, target: str) -> bool:
    s = glued(source)
    return any(c != ("", "") and c not in s.get(v, set()) for v, cs in glued(target).items() for c in cs)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="prod-qwen38-deepseek,prod-nemoultra-deepseek,prod-qwen38-rest")
    parser.add_argument("--repairs", default="")
    parser.add_argument("--out", default="final")
    args = parser.parse_args()
    records, seen = [], set()
    for run in args.runs.split(","):
        d = BASE / run
        api = {r["id"]: r for r in load(d / "second_judge_or.jsonl")}
        local = {r["id"]: r for r in load(d / "second_judge.jsonl")}
        for r in load(d / "accepted.jsonl"):
            if r["id"] in seen:
                raise ValueError(f"passage {r['id']} accepted twice")
            if r["family"] == "rust_book":
                raise ValueError("the Rust Book is the untouched holdout")
            seen.add(r["id"])
            a, l = api.get(r["id"]), local.get(r["id"])
            r["review_flags"] = {**r.get("review_flags", {}),
                                 "second_judge": None if a is None else {"judge": a["judge"], "differences": a["differences"]},
                                 "local_judge": None if l is None else {"judge": l["judge"], "differences": l["differences"]}}
            r["run"] = run
            records.append(r)
    same, moved, applied, index = 0, Counter(), Counter(), {r["id"]: k for k, r in enumerate(records)}
    for run in filter(None, args.repairs.split(",")):    # in order: a later run repairs the current text
        for r in load(BASE / run / "accepted.jsonl"):
            if r["id"] not in index:
                raise ValueError(f"repair of passage {r['id']} that is not in the accepted set")
            old = records[index[r["id"]]]
            if r["repair_of"]["ste"] != old["ste"]:
                raise ValueError(f"repair of {r['id']} in {run} is not of the current text")
            if r["ste"] == old["ste"]:
                same += 1
                continue
            if literal_moved(r["text"], r["ste"]) and not literal_moved(r["text"], old["ste"]):
                moved[run] += 1
                continue
            r["review_flags"] = {**r.get("review_flags", {}),
                                 "second_judge": {"judge": r["judge2_model"], "differences": r["meaning_check2"]},
                                 "local_judge": None}
            r["run"] = run
            r["repaired_from"] = {x: old[x] for x in ("ste", "teacher_model", "run", "review_flags")}
            if "repaired_from" in old:
                r["repaired_from"]["repaired_from"] = old["repaired_from"]
            del r["repair_of"]
            records[index[r["id"]]] = r
            applied[run] += 1
    out = BASE / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "accepted.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flag = lambda r, k: r["review_flags"][k] is not None and (r["review_flags"][k]["differences"] is None
                                                              or bool(r["review_flags"][k]["differences"]))
    summary = {"schema": "asd-ste100-doc-teacher-v5-final", "review_status": "teacher_candidate_owner_unverified",
               "runs": args.runs.split(","), "repair_runs": list(filter(None, args.repairs.split(","))),
               "accepted": len(records), "repaired": sum("repaired_from" in r for r in records),
               "repairs_applied": dict(applied), "repair_same_text": same, "repair_literal_moved": dict(moved),
               "by_teacher": dict(Counter(r["teacher_model"] for r in records)),
               "by_family": dict(Counter(r["family"] for r in records)),
               "by_mode": dict(Counter(r["mode"] for r in records)),
               "attempts": dict(Counter(r["attempts"] for r in records)),
               "second_judge_checked": sum(r["review_flags"]["second_judge"] is not None for r in records),
               "second_judge_flagged": sum(flag(r, "second_judge") for r in records),
               "local_judge_checked": sum(r["review_flags"]["local_judge"] is not None for r in records),
               "local_judge_flagged": sum(flag(r, "local_judge") for r in records),
               "dropped_words_flagged": sum(bool(r["review_flags"].get("dropped_words")) for r in records),
               "accepted_sha256": hashlib.sha256((out / "accepted.jsonl").read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
