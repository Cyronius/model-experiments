import hashlib
import json
import sqlite3

import pytest

from curator import cite_claim, export_sqlite, fingerprint, parse_output, schema_errors, score_extractions
from ingest import chunks, ingest


def fixture():
    text = "# Test observation\nOn device Cedar, kernel ash worked with stack 0x1800.\nCompiler 1.3 was not tested.\n"
    source = {"path": "synthetic://test", "sha256": hashlib.sha256(text.encode()).hexdigest(),
              "context": {}, "upstream_reference_text": "", "upstream_references_verified": False}
    case = {"id": "test", "source_key": "test", "text": text, "source_start": 0,
            "source_end": len(text), "synthetic": True}
    claim = {"statement": "Kernel ash worked with stack 0x1800 on device Cedar.",
             "quote": "On device Cedar, kernel ash worked with stack 0x1800.",
             "conditions": ["device Cedar", "kernel ash"]}
    return text, source, case, claim


def test_exact_citation_and_stable_identity():
    text, source, case, claim = fixture()
    a = cite_claim(claim, case, source, text, "model-A")
    b = cite_claim(claim, case, source, text, "model-B")
    assert a["citation_valid"]
    assert a["id"] == b["id"]
    span = a["source_spans"][0]
    assert span["line_start"] == span["line_end"] == 2
    assert text[span["char_start"]:span["char_end"]] == claim["quote"]
    assert a["review_status"] == "proposed"
    assert not a["semantic_support_verified"]


def test_fabricated_quote_or_condition_is_quarantined():
    text, source, case, claim = fixture()
    bad = dict(claim, quote="Kernel ash failed.")
    assert not cite_claim(bad, case, source, text, "run")["citation_valid"]
    bad = dict(claim, conditions=["driver 9.9"])
    assert not cite_claim(bad, case, source, text, "run")["citation_valid"]
    with pytest.raises(ValueError, match="hash mismatch"):
        cite_claim(claim, case, source, text + "Changed", "run")


def test_cache_identity_changes_with_source_prompt_and_model():
    job = {"source": "a", "prompt": "v1", "model": "r1"}
    assert fingerprint(job) == fingerprint(dict(reversed(list(job.items()))))
    for key in job:
        assert fingerprint(dict(job, **{key: "changed"})) != fingerprint(job)


def test_coverage_does_not_treat_correct_quote_as_correct_statement(tmp_path):
    text, source, case, claim = fixture()
    source["snapshot"] = "source.txt"
    (tmp_path / "source.txt").write_text(text)
    (tmp_path / "manifest.json").write_text(json.dumps({"passages": [case], "sources": {"test": source}}))
    (tmp_path / "gold.json").write_text(json.dumps({"test": [{"id": "works", "patterns": ["worked"], "scope_patterns": ["cedar"]}]}))
    # Verbatim evidence can accompany a semantically false extraction.
    claim["statement"] = "Kernel ash failed with stack 0x1800 on device Cedar."
    row = {"id": "test", "task": "extraction", "output": json.dumps({"claims": [claim]}),
           "output_tokens": 20, "truncated": False, "cache_key": "run"}
    result, records = score_extractions(tmp_path, [row])
    assert records[0]["citation_valid"]
    assert result["totals"]["anchor_matches"] == 0


def test_outcome_refinement_does_not_count_workers_as_worked(tmp_path):
    text, source, case, claim = fixture()
    source["snapshot"] = "source.txt"
    (tmp_path / "source.txt").write_text(text)
    (tmp_path / "manifest.json").write_text(json.dumps({"passages": [case], "sources": {"test": source}}))
    target = {"test": [{"id": "working", "patterns": ["0x1800", "work|succeed|correct|resolve"], "scope_patterns": []}]}
    (tmp_path / "gold.json").write_text(json.dumps(target))
    target["test"][0]["patterns"][1] = r"\b(?:works?|worked|working|succeeds?|succeeded|successful|correct|resolves?|resolved)\b"
    (tmp_path / "gold-refined.json").write_text(json.dumps(target))
    claim["statement"] = "Workers using stack 0x1800 experienced NaN."
    row = {"id": "test", "task": "extraction", "output": json.dumps({"claims": [claim]}),
           "output_tokens": 20, "truncated": False, "cache_key": "run"}
    original, _ = score_extractions(tmp_path, [row])
    corrected, _ = score_extractions(tmp_path, [row], "gold-refined.json")
    assert original["totals"]["anchor_matches"] == 1
    assert corrected["totals"]["anchor_matches"] == 0


def test_sqlite_export_excludes_invalid_evidence(tmp_path):
    text, source, case, claim = fixture()
    good = cite_claim(claim, case, source, text, "run")
    bad = cite_claim(dict(claim, quote="Nonexistent observation."), case, source, text, "run")
    p = tmp_path / "claims.sqlite"
    export_sqlite(p, [good, bad], {"sources": {"test": source}})
    with sqlite3.connect(p) as db:
        assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1


def test_schema_and_json_errors_are_not_silently_repaired():
    assert parse_output('```json\n{"claims": []}\n```') == {"claims": []}
    assert not schema_errors({"claims": []})
    assert schema_errors({"claims": [{"statement": "x"}]})
    with pytest.raises(ValueError):
        parse_output('{"claims": [')


def test_ingestion_preserves_bytes_and_fenced_blocks(tmp_path):
    doc = "# Header\r\n\r\n```cpp\r\nline 1\r\n\r\nline 2\r\n```\r\n\r\nFinal paragraph.\r\n"
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "source.md").write_bytes(doc.encode())
    out = tmp_path / "ingested"
    ingest(root, ["source.md"], out, target=18)
    m = json.loads((out / "manifest.json").read_text())
    source = next(iter(m["sources"].values()))
    assert (out / source["snapshot"]).read_bytes() == doc.encode()
    assert "".join(c["excerpt"] for c in m["passages"]) == doc
    assert all(c["excerpt"].count("```") in (0, 2) for c in m["passages"])
    assert json.loads((out / "gold.json").read_text()) == {}
