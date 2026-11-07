"""Independent full audit of pinned source files, license notices, and splits."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-teacher-pilot-v1"


def read_rows(name):
    return [json.loads(line) for line in (OUT / name).open()]


def main():
    manifest = json.loads((OUT / "source-manifest.json").read_text())
    mapping = {}
    for record in manifest["files"]:
        path = OUT / "sources" / record["family"] / record["path"]
        body = path.read_bytes()
        assert len(body) == record["bytes"]
        assert hashlib.sha256(body).hexdigest() == record["sha256"]
        mapping[(record["family"], record["path"])] = record
    for family, spec in manifest["families"].items():
        assert set(spec["source_paths"] + spec["license_paths"]) == {
            path for (record_family, path) in mapping if record_family == family}
        for path in spec["license_paths"]:
            assert mapping[(family, path)]["role"] == "license_notice"
    candidates = read_rows("candidate_sources.jsonl")
    holdout = read_rows("untouched_holdout.jsonl")
    selected = read_rows("teacher_inputs.jsonl")
    controls = read_rows("unchanged_controls.jsonl")
    all_rows = candidates + holdout
    assert len({row["id"] for row in all_rows}) == len(all_rows)
    assert {row["family"] for row in holdout} == {"rust_book"}
    assert all(row["split"] == "candidate_train" for row in candidates)
    assert all(row["split"] == "untouched_holdout" for row in holdout)
    assert {row["id"] for row in selected + controls} <= {row["id"] for row in candidates}
    assert not {row["id"] for row in selected} & {row["id"] for row in controls}
    assert len(selected) == 36 and len(controls) == 12
    assert all(row["checker_error_rules"] for row in selected)
    assert all(not row["checker_error_rules"] for row in controls)
    for row in all_rows:
        assert row["source_sha256"] == mapping[(row["family"], row["doc_path"])]["sha256"]
        assert row["source_url"] == mapping[(row["family"], row["doc_path"])]["url"]
    result = {"passed": True, "source_files": sum(v["role"] == "source" for v in manifest["files"]),
              "license_notice_files": sum(v["role"] == "license_notice" for v in manifest["files"]),
              "candidate_source_passages": len(candidates), "untouched_holdout_passages": len(holdout),
              "teacher_inputs": len(selected), "unchanged_controls": len(controls),
              "selected_by_family": dict(Counter(row["family"] for row in selected)),
              "source_manifest_sha256": hashlib.sha256((OUT / "source-manifest.json").read_bytes()).hexdigest(),
              "selection_plan_sha256": hashlib.sha256((OUT / "synthesis-plan.json").read_bytes()).hexdigest()}
    (OUT / "source-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
