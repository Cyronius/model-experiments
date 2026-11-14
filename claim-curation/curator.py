"""Model-independent citation checks, stable records, metrics, and SQLite export.

An exact source match is a citation check, never proof of semantic entailment.
All exported claims remain proposed until reviewed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

RELATIONS = {"contradicts", "supports", "different_scope", "insufficient_information"}


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def parse_output(text):
    """Allow one Markdown JSON fence; never repair malformed model output."""
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, count=1)[:-3].strip()
    return json.loads(s)


def schema_errors(payload):
    if not isinstance(payload, dict) or set(payload) != {"claims"} or not isinstance(payload["claims"], list):
        return ["Expected exactly one claims array"]
    errors = []
    for i, claim in enumerate(payload["claims"]):
        if not isinstance(claim, dict) or set(claim) != {"statement", "quote", "conditions"}:
            errors.append(f"claim {i}: fields do not match the schema")
            continue
        for key in ("statement", "quote"):
            if not isinstance(claim[key], str) or not claim[key].strip():
                errors.append(f"claim {i}: {key} must be a nonempty string")
        cond = claim["conditions"]
        if not isinstance(cond, list) or any(not isinstance(c, str) or not c.strip() for c in cond):
            errors.append(f"claim {i}: conditions must be nonempty strings")
    return errors


def cite_claim(claim, case, source, snapshot, run_key):
    if hashlib.sha256(snapshot.encode()).hexdigest() != source["sha256"]:
        raise ValueError("Source snapshot hash mismatch")
    quote = claim["quote"]
    errors = []
    if quote not in case["text"]:
        errors.append("quote is not verbatim in supplied text")
    missing = [c for c in claim["conditions"] if c not in case["text"]]
    if missing:
        errors.append("condition fragments are not verbatim: " + repr(missing))
    spans = []
    for match in re.finditer(re.escape(quote), snapshot):
        a, b = match.span()
        in_excerpt = case["source_start"] <= a and b <= case["source_end"]
        in_header = a < snapshot.find("---", 3) + 3 if snapshot.startswith("---") else False
        if in_excerpt or in_header:
            spans.append({"char_start": a, "char_end": b,
                          "line_start": snapshot.count("\n", 0, a) + 1,
                          "line_end": snapshot.count("\n", 0, max(a, b - 1)) + 1})
    if not spans:
        errors.append("quote has no allowed original-source span")
    evidence_id = "E-" + fingerprint([source["sha256"], spans, quote])[:20]
    claim_id = "C-" + fingerprint([evidence_id, " ".join(claim["statement"].split()), claim["conditions"]])[:20]
    return {"id": claim_id, "evidence_id": evidence_id, **claim,
            "passage_id": case["id"], "source_key": case["source_key"],
            "synthetic": case["synthetic"], "source_path": source["path"],
            "source_sha256": source["sha256"], "source_spans": spans,
            "source_context": source["context"],
            "upstream_reference_text": source["upstream_reference_text"],
            "upstream_references_verified": source["upstream_references_verified"],
            "review_status": "proposed", "semantic_support_verified": False,
            "citation_valid": not errors, "validation_errors": errors, "run_key": run_key}


def score_extractions(data: Path, raw_rows, gold_name="gold.json"):
    manifest = json.loads((data / "manifest.json").read_text())
    gold = json.loads((data / gold_name).read_text())
    cases = {c["id"]: c for c in manifest["passages"]}
    records, cases_report = [], []
    for row in raw_rows:
        if row["task"] != "extraction":
            continue
        case = cases[row["id"]]
        try:
            payload = parse_output(row["output"])
            errors = schema_errors(payload)
        except (ValueError, TypeError) as exc:
            payload, errors = {"claims": []}, [str(exc)]
        local = []
        if not errors:
            source = manifest["sources"][case["source_key"]]
            snapshot = (data / source["snapshot"]).read_bytes().decode("utf-8")
            local = [cite_claim(c, case, source, snapshot, row["cache_key"]) for c in payload["claims"]]
            records.extend(local)
        targets = []
        for expected in gold.get(row["id"], []):
            matches, scoped, grounded = [], [], []
            for c in local:
                text = c["statement"].lower().replace("−", "-").replace("–", "-")
                if all(re.search(pattern, text) for pattern in expected["patterns"]):
                    matches.append(c["id"])
                    context = text + " " + " ".join(c["conditions"]).lower()
                    if all(re.search(pattern, context) for pattern in expected["scope_patterns"]):
                        scoped.append(c["id"])
                        if c["citation_valid"]:
                            grounded.append(c["id"])
            targets.append({**expected, "matched_claims": matches, "scoped_claims": scoped,
                            "scoped_and_cited_claims": grounded})
        overmerged = []
        for c in local:
            covered = [t["id"] for t in targets if c["id"] in t["matched_claims"]]
            if len(covered) > 1:
                overmerged.append({"claim_id": c["id"], "anchors": covered})
        cases_report.append({"id": row["id"], "synthetic": case["synthetic"],
                             "schema_errors": errors, "output_tokens": row["output_tokens"],
                             "truncated": row["truncated"], "claims": len(local), "targets": targets,
                             "possible_compound_claims": overmerged,
                             "annotation_present": row["id"] in gold,
                             "empty_expected": row["id"] in gold and not gold[row["id"]], "empty_actual": not local})
    totals = {"passages": len(cases_report),
              "schema_valid_passages": sum(not c["schema_errors"] for c in cases_report),
              "claims": len(records), "verbatim_cited_claims": sum(c["citation_valid"] for c in records),
              "gold_anchors": sum(len(c["targets"]) for c in cases_report),
              "anchor_matches": sum(bool(t["matched_claims"]) for c in cases_report for t in c["targets"]),
              "scoped_anchor_matches": sum(bool(t["scoped_claims"]) for c in cases_report for t in c["targets"]),
              "scoped_and_cited_matches": sum(bool(t["scoped_and_cited_claims"]) for c in cases_report for t in c["targets"]),
              "empty_control_passages": sum(c["empty_expected"] for c in cases_report),
              "empty_control_correct": sum(c["empty_expected"] and c["empty_actual"] and not c["schema_errors"] for c in cases_report),
              "claims_on_empty_control": sum(c["claims"] for c in cases_report if c["empty_expected"]),
              "truncated_passages": sum(c["truncated"] for c in cases_report),
              "possible_compound_claims": sum(len(c["possible_compound_claims"]) for c in cases_report)}
    return {"totals": totals, "passages": cases_report}, records


def score_relations(data: Path, rows):
    golds = {p["id"]: p for p in json.loads((data / "pairs.json").read_text())}
    detail = []
    for row in rows:
        if row["task"] != "relation":
            continue
        expected = golds[row["id"]]
        try:
            result = parse_output(row["output"])
            valid = (isinstance(result, dict) and set(result) == {"relation", "reason"}
                     and isinstance(result["relation"], str) and result["relation"] in RELATIONS
                     and isinstance(result["reason"], str) and bool(result["reason"].strip()))
        except (ValueError, TypeError):
            result, valid = {}, False
        detail.append({"id": row["id"], "synthetic": expected["synthetic"],
                       "expected": expected["gold_relation"], "actual": result.get("relation"),
                       "reason": result.get("reason"), "valid": valid,
                       "correct": valid and result["relation"] == expected["gold_relation"]})
    positives = [d for d in detail if d["actual"] == "contradicts"]
    actual = [d for d in detail if d["expected"] == "contradicts"]
    tp = sum(d["correct"] for d in positives)
    return {"correct": sum(d["correct"] for d in detail), "total": len(detail),
            "valid": sum(d["valid"] for d in detail),
            "contradiction_true_positives": tp, "contradiction_predictions": len(positives),
            "contradictions_in_gold": len(actual), "pairs": detail}


def export_sqlite(path: Path, records, manifest):
    """Rebuild a derived artifact, leaving raw documents and model output intact."""
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
        DROP TABLE IF EXISTS claims;
        DROP TABLE IF EXISTS sources;
        CREATE TABLE sources (source_key TEXT PRIMARY KEY, data_json TEXT NOT NULL);
        CREATE TABLE claims (id TEXT PRIMARY KEY, source_key TEXT NOT NULL,
          evidence_id TEXT NOT NULL, review_status TEXT NOT NULL, data_json TEXT NOT NULL,
          FOREIGN KEY(source_key) REFERENCES sources(source_key));
        CREATE INDEX claims_by_source ON claims(source_key);
        CREATE INDEX claims_by_evidence ON claims(evidence_id);
        """)
        db.executemany("INSERT INTO sources VALUES (?,?)", [(k, json.dumps(v)) for k, v in manifest["sources"].items()])
        unique = {r["id"]: r for r in records if r["citation_valid"]}
        db.executemany("INSERT INTO claims VALUES (?,?,?,?,?)", [
            (r["id"], r["source_key"], r["evidence_id"], r["review_status"], json.dumps(r)) for r in unique.values()])


def report(data, out):
    rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    extraction, records = score_extractions(data, rows)
    relations = score_relations(data, rows)
    (out / "extraction-scores.json").write_text(json.dumps(extraction, indent=2) + "\n")
    if (data / "gold-refined.json").exists():
        refined, _ = score_extractions(data, rows, "gold-refined.json")
        (out / "extraction-scores-refined.json").write_text(json.dumps(refined, indent=2) + "\n")
    (out / "relation-scores.json").write_text(json.dumps(relations, indent=2) + "\n")
    (out / "claims.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "quarantine.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records if not r["citation_valid"]))
    manifest = json.loads((data / "manifest.json").read_text())
    export_sqlite(out / "claims.sqlite", records, manifest)
    review = ["# Proposed claims for review", "", "Exact citations are checked mechanically. Semantic support has not been certified.", ""]
    for record in records:
        snapshot = manifest["sources"][record["source_key"]]["snapshot"]
        positions = ", ".join(f'lines {s["line_start"]}-{s["line_end"]}' for s in record["source_spans"])
        review += [f'## {record["id"]}', "", record["statement"], "",
                   f'Conditions: {json.dumps(record["conditions"], ensure_ascii=False)}', "",
                   f'Source: [{record["source_path"]}](../../data/{snapshot}), {positions or "unresolved"}.', "",
                   "```text", record["quote"], "```", "",
                   f'Citation valid: {record["citation_valid"]}. Review status: proposed.']
        if record["validation_errors"]:
            review += ["", "Validation: " + "; ".join(record["validation_errors"])]
        review.append("")
    (out / "review.md").write_text("\n".join(review) + "\n")
    print(json.dumps({"extraction": extraction["totals"], "relation_correct": relations["correct"], "relation_total": relations["total"]}, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    report(a.data, a.out)
