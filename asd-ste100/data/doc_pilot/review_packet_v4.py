"""Render the first N v4 teacher results for source-by-source semantic review.

Each passage shows the final rewrite and, when it differs, the first rewrite
that passed the gates (what a run without the meaning check would accept).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from review_packet import load

OUT = Path(__file__).resolve().parents[1] / "out/doc-teacher-pilot-v4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=132)
    args = parser.parse_args()
    first = load(OUT, "teacher_inputs.jsonl")[:args.limit]
    results = {row["id"]: row for name in ("accepted.jsonl", "rejected.jsonl") for row in load(OUT, name)}
    if any(row["id"] not in results for row in first):
        raise ValueError(f"first {args.limit} teacher results are incomplete")

    lines = [f"# ASD-STE100 documentation teacher pilot v4: first {len(first)}", "",
             "Teacher outputs are candidates. Review each for preserved technical meaning. Passages 1-96 are new; "
             "97-132 are the v1 inputs that the v4 fixes were designed on.", ""]
    for index, source in enumerate(first, 1):
        result, gates_only = results[source["id"]], results[source["id"]]["gates_only"]
        first_pass = f"attempt {gates_only['attempt']}" if gates_only else "none"
        lines += [f"## {index}. {source['family']} — {source['doc_path']} ({source['pilot_set']})", "",
                  f"Source: {source['source_url']}", "",
                  f"Result: {'accepted' if not result['issues'] else 'rejected'}; attempts: {result['attempts']}; "
                  f"first gate pass: {first_pass}", "",
                  "Source passage:", "", "~~~text", source["text"], "~~~", "",
                  "Final rewrite:", "", "~~~text", result["ste"] or "(no parsed rewrite)", "~~~", ""]
        if gates_only and gates_only["ste"] != result["ste"]:
            lines += ["First rewrite that passed the gates:", "", "~~~text", gates_only["ste"], "~~~", ""]
        lines += ["Deterministic issues: " + (", ".join(result["issues"]) or "none"), "",
                  "Teacher meaning check: " + ("; ".join(result["meaning_check"] or []) or
                                               ("no differences" if result["meaning_check"] is not None else "not run")), "",
                  "Source words not in the rewrite: " + (", ".join(result["review_flags"]["dropped_words"]) or "none"), "",
                  "Semantic verdict: pending", ""]
    (OUT / f"first-{len(first)}-review.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
