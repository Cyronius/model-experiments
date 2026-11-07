"""Freeze a small, source-family-separated teacher pilot before inference."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-teacher-pilot-v1"
sys.path.insert(0, str(ROOT / "checker"))
from checker import Lexicon, check, errors  # noqa: E402


def key(row):
    return hashlib.sha256(("doc-teacher-pilot-v1/" + row["id"]).encode()).digest()


def self_contained(row):
    text = row["text"]
    # The teacher sees one passage at a time. Drop passages whose opening or
    # internal reference needs prose, examples, or a heading outside it.
    return not re.search(
        r"^(?:These|This|Those|Such|So|Of course|In particular|However|Note that|Alternatively)\b"
        r"|\bboth messages\b|\bthese messages\b"
        r"|\bsame as before\b|^The constructor\b|^The URL column\b"
        r"|\b(?:such (?:a|an|slots|operations|pages)|these reasons|the other two approaches|the following subsections|the previous version)\b",
        text, re.I,
    )


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    manifest = json.loads((OUT / "source-manifest.json").read_text())
    assert manifest["families"]["rust_book"]["split"] == "untouched_holdout"
    rows = [json.loads(line) for line in (OUT / "candidate_sources.jsonl").open()]
    assert len(rows) >= 100 and not any(row["family"] == "rust_book" for row in rows)
    lex = Lexicon()
    for row in rows:
        row["checker_error_rules"] = [v.rule for v in errors(check(row["text"], lex, row["mode"]))]
        row["word_count"] = len(row["text"].split())
    by_doc = {}
    for row in rows:
        by_doc.setdefault(row["doc_path"], []).append(row)
    selected = []
    quotas = {
        "Doc/library/asyncio-task.rst": 4,
        "Doc/howto/logging.rst": 8,
        "doc/src/sgml/maintenance.sgml": 8,
        "doc/src/sgml/backup.sgml": 4,
    }
    for path, limit in quotas.items():
        pool = [r for r in by_doc[path] if r["checker_error_rules"] and 15 <= r["word_count"] <= 55 and self_contained(r)]
        pool.sort(key=key)
        if len(pool) < limit:
            raise ValueError(f"too few rewrite candidates in {path}: {len(pool)}")
        selected += pool[:limit]
    curl = {}
    for row in rows:
        if row["family"] == "curl_cli" and row["checker_error_rules"] and 15 <= row["word_count"] <= 55 and self_contained(row):
            curl.setdefault(row["doc_path"], []).append(row)
    for values in curl.values():
        values.sort(key=key)
    curl_docs = sorted(curl, key=lambda path: hashlib.sha256(("curl-doc/" + path).encode()).digest())
    if len(curl_docs) < 12:
        raise ValueError(f"too few curl option documents: {len(curl_docs)}")
    selected += [curl[path][0] for path in curl_docs[:12]]
    assert len(selected) == 36 and len({r["id"] for r in selected}) == 36
    selected.sort(key=lambda r: (r["family"], key(r)))
    controls = []
    for family, limit in (("python_docs", 5), ("postgres_docs", 2), ("curl_cli", 5)):
        pool = [r for r in rows if r["family"] == family and not r["checker_error_rules"]
                and 12 <= r["word_count"] <= 55 and self_contained(r)
                and r["id"] not in {s["id"] for s in selected}]
        pool.sort(key=key)
        if len(pool) < limit:
            raise ValueError(f"too few clean controls in {family}: {len(pool)}")
        controls += pool[:limit]
    controls.sort(key=lambda r: (r["family"], key(r)))
    write("teacher_inputs.jsonl", selected)
    write("unchanged_controls.jsonl", controls)
    plan = {
        "schema": "asd-ste100-doc-teacher-plan-v1",
        "source_manifest_sha256": hashlib.sha256((OUT / "source-manifest.json").read_bytes()).hexdigest(),
        "selection_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": {"candidate_train": ["python_docs", "postgres_docs", "curl_cli"],
                   "untouched_holdout": ["rust_book"]},
        "teacher_pairs_planned": 36,
        "unchanged_controls_planned": 12,
        "teacher_by_family": dict(Counter(row["family"] for row in selected)),
        "controls_by_family": dict(Counter(row["family"] for row in controls)),
        "selection": "SHA sampling from checker-flagged 15-55 word self-contained passages, with document quotas; controls are checker-clean sources. All are candidate labels, not gold.",
        "limits": ["Rust holdout sources remain untouched and are not sent to the teacher.",
                   "A checker-clean source is not necessarily valid ASD-STE100 or meaning gold.",
                   "Source licenses and notices remain beside exact raw files."],
    }
    (OUT / "synthesis-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("teacher_by_family", "controls_by_family")}))


if __name__ == "__main__":
    main()
