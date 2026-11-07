"""Freeze the v4 teacher inputs: 96 new passages from the frozen v1 files, then the 36 v1 inputs.

v1 kept only the first `per_file` passages of each source file, by id. v4 runs
the same parsers and segmenter on the same frozen bytes (hash-checked against
the v1 manifest) and draws passages that no earlier pilot used as an input, a
control or a prompt example. The selection rule is the v1 rule: checker-flagged,
15-55 words, self-contained, SHA-ordered within document quotas. The 36 v1
inputs follow in the v2 order, so v3 and v4 compare on the same passages.
The Rust family stays an untouched holdout and is not read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/out/doc-teacher-pilot-v1"
OUT = ROOT / "data/out/doc-teacher-pilot-v4"
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "checker"), str(ROOT / "tokenizer"), str(Path(__file__).parent)]
from checker import Lexicon, check, errors  # noqa: E402
from select_sources import self_contained  # noqa: E402
from synthesize_v2 import first_rows  # noqa: E402

# tokenizer/prepare.py shadows this directory's prepare.py on the path.
_spec = importlib.util.spec_from_file_location("doc_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)
FAMILIES, mode, segments, sha = prepare.FAMILIES, prepare.mode, prepare.segments, prepare.sha
PARSERS = {".rst": prepare.rst_paragraphs, ".sgml": prepare.sgml_paragraphs, ".md": prepare.md_paragraphs}

QUOTAS = {
    "curl_cli": {"*": 36},
    "postgres_docs": {"doc/src/sgml/maintenance.sgml": 18, "doc/src/sgml/backup.sgml": 18},
    "python_docs": {"Doc/howto/logging.rst": 24},
}
CURL_PER_FILE = 4


def key(row):
    return hashlib.sha256(("doc-teacher-pilot-v4/" + row["id"]).encode()).digest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def example_ids() -> set[str]:
    from ste_context import EXAMPLES
    from ste_context_v3 import JUDGE_EXAMPLES
    return {e[0] for e in EXAMPLES} | {e[0] for e in JUDGE_EXAMPLES}


def passages(manifest: dict) -> list[dict]:
    """Every passage of the frozen candidate_train files, as prepare.py extracts them."""
    files = {(f["family"], f["path"]): f for f in manifest["files"]}
    out = []
    for family, spec in FAMILIES.items():
        if manifest["families"][family]["split"] != "candidate_train":
            continue
        for path in spec["paths"]:
            record = files[(family, path)]
            body = (V1 / "sources" / family / path).read_bytes()
            if sha(body) != record["sha256"]:
                raise ValueError(f"frozen source changed: {family}/{path}")
            for index, paragraph in enumerate(PARSERS[Path(path).suffix](body.decode("utf-8"))):
                for part in segments(paragraph):
                    out.append(dict(id=sha((family + "\n" + path + "\n" + part).encode()), family=family,
                                    split="candidate_train", doc_path=path, source_url=record["url"],
                                    source_sha256=record["sha256"], passage_index=index, text=part, mode=mode(part)))
    return out


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    manifest = json.loads((V1 / "source-manifest.json").read_text())
    assert manifest["families"]["rust_book"]["split"] == "untouched_holdout"
    old = [json.loads(l) for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl")
           for l in (V1 / name).read_text().splitlines()]
    candidates = {json.loads(l)["id"]: json.loads(l) for l in (V1 / "candidate_sources.jsonl").read_text().splitlines()}
    used_ids = {r["id"] for r in old} | example_ids()
    used_text = {normalized(r["text"]) for r in old} | {normalized(candidates[i]["text"]) for i in example_ids()}
    lex = Lexicon()
    pool, seen = [], set()
    for row in sorted(passages(manifest), key=lambda r: r["id"]):
        text = normalized(row["text"])
        if row["id"] in used_ids or text in used_text or text in seen:
            continue
        seen.add(text)
        row["checker_error_rules"] = [v.rule for v in errors(check(row["text"], lex, row["mode"]))]
        row["word_count"] = len(row["text"].split())
        if row["checker_error_rules"] and 15 <= row["word_count"] <= 55 and self_contained(row):
            pool.append(row)
    pool.sort(key=key)
    selected = {}
    for family, quotas in QUOTAS.items():
        rows = []
        for path, limit in quotas.items():
            matches = [r for r in pool if r["family"] == family and path in ("*", r["doc_path"])]
            if path == "*":
                capped, per_file = [], Counter()
                for r in matches:
                    per_file[r["doc_path"]] += 1
                    if per_file[r["doc_path"]] <= CURL_PER_FILE:
                        capped.append(r)
                matches = capped
            if len(matches) < limit:
                raise ValueError(f"too few new passages in {family}/{path}: {len(matches)}")
            rows += matches[:limit]
        selected[family] = rows
    new = [dict(r, pilot_set="v4_new") for position in range(max(map(len, selected.values())))
           for family in ("curl_cli", "postgres_docs", "python_docs") for r in selected[family][position:position + 1]]
    repeat = [dict(r, pilot_set="v1_inputs") for r in first_rows(36)]
    assert len({r["id"] for r in new + repeat}) == len(new + repeat) == 132
    OUT.mkdir(parents=True, exist_ok=True)
    write("teacher_inputs.jsonl", new + repeat)
    plan = {
        "schema": "asd-ste100-doc-teacher-plan-v4",
        "source_manifest_sha256": hashlib.sha256((V1 / "source-manifest.json").read_bytes()).hexdigest(),
        "selection_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": {"candidate_train": ["python_docs", "postgres_docs", "curl_cli"], "untouched_holdout": ["rust_book"]},
        "new_by_family": dict(Counter(r["family"] for r in new)),
        "new_by_document": dict(Counter(r["doc_path"] for r in new)),
        "eligible_new_by_family": dict(Counter(r["family"] for r in pool)),
        "v1_inputs": len(repeat),
        "order": "96 new passages, one from each family in turn, then the 36 v1 inputs in the v2 order.",
        "selection": "Same frozen v1 files, parsers and segmenter; passages not used before as inputs, controls or prompt "
                     "examples; checker-flagged, 15-55 words, self-contained; SHA order within document quotas, "
                     f"at most {CURL_PER_FILE} from each curl option file.",
        "limits": ["New passages come from the same documents as the v1 inputs, not from new documents.",
                   "Rust holdout sources remain untouched and are not sent to the teacher.",
                   "All outputs are candidate labels, not gold."],
    }
    (OUT / "synthesis-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("new_by_family", "eligible_new_by_family", "v1_inputs")}))


if __name__ == "__main__":
    main()
