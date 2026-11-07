"""A large passage pool for a scaled teacher run: whole documentation trees, same rules as v1-v4.

Downloads each candidate_train project's source tarball at the commit that the v1
manifest pins, keeps the documentation files (Python Doc/**/*.rst, PostgreSQL
doc/src/sgml/*.sgml, curl docs/**/*.md), and runs the v1 parsers, segmenter and
selection rule on them (checker-flagged, 15-55 words, self-contained). Every passage
that a pilot used as an input, a control or a prompt example is left out, by id and by
text. The Rust Book stays an untouched holdout. Writes data/out/doc-pool-v5/:
pool.jsonl (the v4 input schema), files.json (path, sha256, license of each file),
summary.json. The tarballs are scratch and are deleted after the run.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from select_sources_v4 import (FAMILIES, PARSERS, V1, example_ids, mode, normalized, segments,  # noqa: E402
                               self_contained, sha)
from checker import Lexicon, check, errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-pool-v5"
DOCS = {"python_docs": ("Doc/", ".rst"), "postgres_docs": ("doc/src/sgml/", ".sgml"), "curl_cli": ("docs/", ".md")}


def tarball(repo: str, commit: str) -> bytes:
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilots = [json.loads(l) for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl",
                                         "candidate_sources.jsonl")
              for l in (V1 / name).read_text().splitlines()]
    pilots += [json.loads(l) for l in (ROOT / "data/out/doc-teacher-pilot-v4/teacher_inputs.jsonl").read_text().splitlines()]
    used_ids = {r["id"] for r in pilots} | example_ids()
    used_text = {normalized(r["text"]) for r in pilots}
    lex = Lexicon()
    pool, files, seen, counts = [], [], set(), Counter()
    for family, (prefix, suffix) in DOCS.items():
        spec = FAMILIES[family]
        assert spec["split"] == "candidate_train"
        data = tarball(spec["repo"], spec["commit"])
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                path = member.name.split("/", 1)[1] if "/" in member.name else member.name
                if not (member.isfile() and path.startswith(prefix) and path.endswith(suffix)):
                    continue
                body = tar.extractfile(member).read()
                try:
                    text = body.decode("utf-8")
                except UnicodeDecodeError:
                    counts[family + ":undecodable"] += 1
                    continue
                url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
                files.append({"family": family, "path": path, "url": url, "sha256": sha(body), "license": spec["license"]})
                for index, paragraph in enumerate(PARSERS[suffix](text)):
                    for part in segments(paragraph):
                        counts[family + ":segments"] += 1
                        key = normalized(part)
                        row = dict(id=sha((family + "\n" + path + "\n" + part).encode()), family=family,
                                   split="candidate_train", doc_path=path, source_url=url, source_sha256=sha(body),
                                   passage_index=index, text=part, mode=mode(part))
                        if row["id"] in used_ids or key in used_text or key in seen:
                            continue
                        seen.add(key)
                        words = len(part.split())
                        if not 15 <= words <= 55 or not self_contained(row):
                            continue
                        rules = [v.rule for v in errors(check(part, lex, row["mode"]))]
                        if rules:
                            pool.append({**row, "checker_error_rules": rules, "word_count": words})
        print(json.dumps({"family": family, "files": sum(1 for f in files if f["family"] == family),
                          "pool": sum(1 for r in pool if r["family"] == family)}), flush=True)
    pool.sort(key=lambda r: r["id"])
    with (OUT / "pool.jsonl").open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "files.json").write_text(json.dumps(files, indent=1) + "\n")
    summary = {"schema": "asd-ste100-doc-pool-v5", "pool": len(pool), "by_family": dict(Counter(r["family"] for r in pool)),
               "by_mode": dict(Counter(r["mode"] for r in pool)), "files": len(files), "counts": dict(counts),
               "commits": {f: FAMILIES[f]["commit"] for f in DOCS}, "holdout": "rust_book (not read)"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
