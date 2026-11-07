"""Render the first N teacher results (default 12) for source-by-source semantic review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

V1 = Path(__file__).resolve().parents[1] / "out/doc-teacher-pilot-v1"


def load(out: Path, name: str) -> list[dict]:
    path = out / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=V1, help="pilot output directory; inputs always come from v1")
    parser.add_argument("--limit", type=int, default=12, help="a multiple of 3: the same number from each family")
    args = parser.parse_args()
    out, limit = args.out, args.limit
    if limit % 3 or not 3 <= limit <= 36:
        raise ValueError("limit must be a multiple of 3 from 3 to 36")
    rows = load(V1, "teacher_inputs.jsonl")
    by_family = {family: [row for row in rows if row["family"] == family]
                 for family in ("curl_cli", "postgres_docs", "python_docs")}
    first = [by_family[family][position] for position in range(limit // 3) for family in by_family]
    results = {row["id"]: row for name in ("accepted.jsonl", "rejected.jsonl") for row in load(out, name)}
    if any(row["id"] not in results for row in first):
        raise ValueError(f"first {limit} teacher results are incomplete")

    lines = [f"# ASD-STE100 documentation teacher pilot: first {limit}", "",
             "Teacher outputs are candidates. Review each for preserved technical meaning.", ""]
    for index, source in enumerate(first, 1):
        result = results[source["id"]]
        lines += [f"## {index}. {source['family']} — {source['doc_path']}", "",
                  f"Source: {source['source_url']}", "",
                  f"Result: {'passed deterministic gates' if not result['issues'] else 'rejected'}; "
                  f"attempts: {result['attempts']}", "",
                  "Source passage:", "", "~~~text", source["text"], "~~~", "",
                  "Teacher rewrite:", "", "~~~text", result["ste"] or "(no parsed rewrite)", "~~~", "",
                  "Deterministic issues: " + (", ".join(result["issues"]) or "none"), ""]
        if "meaning_check" in result:
            lines += ["Teacher meaning check: " + ("; ".join(result["meaning_check"] or []) or
                                                   ("no differences" if result["meaning_check"] is not None else "not run")), "",
                      "Source words not in the rewrite: " + (", ".join(result["review_flags"]["dropped_words"]) or "none"), ""]
        lines += ["Semantic verdict: pending", ""]
    (out / f"first-{limit}-review.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
