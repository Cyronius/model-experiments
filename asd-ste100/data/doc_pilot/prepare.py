"""Fetch four pinned computing-document families for a bounded STE teacher pilot.

The Rust family is held out by project before passage extraction or synthesis.
All fetched source and license bytes, hashes, and URLs remain in the output.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-teacher-pilot-v1"

FAMILIES = {
    "python_docs": {
        "repo": "python/cpython", "branch": "main",
        "commit": "862d5e5ed889fb8a34ccd3106ec5e5b07f8b19f3",
        "commit_date_utc": "2026-10-26T16:38:45Z", "split": "candidate_train",
        "license": "Python Software Foundation License Version 2 (documentation prose)",
        "license_paths": ["LICENSE", "Doc/license.rst"],
        "paths": ["Doc/library/asyncio-task.rst", "Doc/howto/logging.rst"],
        "per_file": 30,
    },
    "postgres_docs": {
        "repo": "postgres/postgres", "branch": "master",
        "commit": "6e9e5681b8a7ff3c5eea09db980e888c8e81ad9c",
        "commit_date_utc": "2026-10-26T17:01:59Z", "split": "candidate_train",
        "license": "PostgreSQL License (explicitly covers documentation)",
        "license_paths": ["COPYRIGHT"],
        "paths": ["doc/src/sgml/maintenance.sgml", "doc/src/sgml/backup.sgml"],
        "per_file": 30,
    },
    "curl_cli": {
        "repo": "curl/curl", "branch": "master",
        "commit": "1f506cdb65e295ffbbf30cb15a33d20104f07c75",
        "commit_date_utc": "2026-10-26T16:18:01Z", "split": "candidate_train",
        "license": "curl license; each selected option file carries SPDX-License-Identifier: curl",
        "license_paths": ["COPYING"],
        "paths": [f"docs/cmdline-opts/{name}.md" for name in (
            "retry", "retry-delay", "retry-max-time", "connect-timeout",
            "max-time", "output", "remote-name", "parallel", "fail",
            "header", "location", "compressed", "json", "data-binary",
            "upload-file", "speed-limit", "speed-time")],
        "per_file": 3,
    },
    "rust_book": {
        "repo": "rust-lang/book", "branch": "main",
        "commit": "1500248d8f230566e4ec9f27fcbb8fe9e2898ab1",
        "commit_date_utc": "2026-09-02T16:04:34Z", "split": "untouched_holdout",
        "license": "MIT OR Apache-2.0, at recipient's option",
        "license_paths": ["COPYRIGHT", "LICENSE-MIT", "LICENSE-APACHE"],
        "paths": ["src/ch16-01-threads.md", "src/ch16-02-message-passing.md"],
        "per_file": 10,
    },
}


def get(repo: str, commit: str, path: str) -> tuple[bytes, str]:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "asd-ste100-doc-pilot/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    if not body or len(body) > 1_000_000:
        raise ValueError(f"unexpected source size for {url}: {len(body)}")
    return body, url


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_markup(value: str) -> str:
    value = re.sub(r":(?:[a-z]+:)?[a-z-]+:`([^`]+)`", r"`\1`", value)
    value = re.sub(r"`([^`<>]+?)\s*<https?://[^>]+>`_", r"\1", value)
    value = re.sub(r"``([^`]+)``", r"`\1`", value)
    value = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*", lambda m: m[1] or m[2], value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,;:.!?])", r"\1", value)
    return value


def rst_paragraphs(raw: str):
    for block in re.split(r"\n\s*\n", raw):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines or any(line.startswith(".. ") or line.startswith("   ") for line in lines):
            continue
        if any(line.lstrip().startswith((">>>", "...", "$ ", ":")) for line in lines):
            continue
        if all(re.fullmatch(r"[-=*#~^\"']+", line.strip()) for line in lines):
            continue
        value = clean_markup(" ".join(lines))
        if "::" in value or ".. _" in value or ".. " in value:
            continue
        yield value


def sgml_paragraphs(raw: str):
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
    raw = re.sub(r"<(programlisting|screen|synopsis|indexterm|literallayout)\b[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    for match in re.finditer(r"<para(?:\s[^>]*)?>(.*?)</para>", raw, flags=re.S | re.I):
        value = match[1]
        if re.search(r"<(programlisting|screen|table|xref)\b", value, re.I):
            continue
        value = re.sub(r"<(literal|command|filename|function|option|varname|parameter|type)\b[^>]*>(.*?)</\1>",
                       lambda m: "`" + re.sub(r"<[^>]+>", "", m[2]) + "`", value, flags=re.S | re.I)
        value = re.sub(r"<xref\b[^>]*/>", " ", value)
        value = re.sub(r"<[^>]+>", " ", value)
        value = clean_markup(html.unescape(value))
        if "&" in value and re.search(r"&[a-zA-Z][\w.-]*;", value):
            continue
        yield value


def md_paragraphs(raw: str):
    raw = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.S)
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
    raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
    for block in re.split(r"\n\s*\n", raw):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or any(line.startswith(("#", "|", "<", "```")) for line in lines):
            continue
        value = " ".join(lines)
        value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
        value = clean_markup(value)
        yield value


def segments(value: str):
    # Keep short, complete 1-2 sentence passages. Avoid code blocks and headings.
    value = value.strip()
    if not value or value.startswith(("* ", "- ", "+ ", "..", ":::")):
        return
    if any(token in value for token in ("<", ">", "\\", "&quot;")):
        return
    if re.search(r"\b(?:Listing|Figure|Chapter|previous section|next section|previous version|as described above|as shown below|favourite beverage)\b", value, re.I):
        return
    if re.match(r"^(?:In this case|In that case),", value, re.I):
        return
    if re.search(r"\b(?:in|at|from|to|see|about)\s*\.|\(\s*see\s*\)", value, re.I):
        return
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", value)
    for sentence in sentences:
        sentence = sentence.strip()
        if re.match(r"^(?:In this case|In that case),", sentence, re.I):
            continue
        if re.search(r"\b(?:Listing|Figure|Chapter|previous section|next section|previous version|as described above|as shown below|favourite beverage)\b", sentence, re.I):
            continue
        n = len(sentence.split())
        if 12 <= n <= 65 and sentence.endswith((".", "?", "!")):
            if sentence.count("`") % 2 or sentence.count('"') % 2:
                continue
            yield sentence


def mode(text: str) -> str:
    return "PROC" if re.match(r"^(?:Use|Run|Set|Copy|Open|Close|Write|Read|Check|Start|Stop|Do|Make|Call|Enter|Select|Specify|Add|Remove|Create|Delete|Put|Keep|Send|Save|Press)\b", text) else "DESC"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / "sources"
    source_dir.mkdir(exist_ok=True)
    manifest = {"schema": "asd-ste100-doc-teacher-sources-v1",
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
                "split_rule": "Entire rust_book family is untouched holdout; all other families candidate_train, assigned before extraction or synthesis.",
                "families": {}, "files": []}
    selected = []
    for family, spec in FAMILIES.items():
        manifest["families"][family] = {k: v for k, v in spec.items() if k not in ("paths", "license_paths")}
        manifest["families"][family]["source_paths"] = spec["paths"]
        manifest["families"][family]["license_paths"] = spec["license_paths"]
        for path in spec["license_paths"] + spec["paths"]:
            body, url = get(spec["repo"], spec["commit"], path)
            if family == "curl_cli" and path in spec["paths"] and b"SPDX-License-Identifier: curl" not in body[:300]:
                raise ValueError("curl file lacks expected SPDX notice: " + path)
            destination = source_dir / family / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            manifest["files"].append(dict(family=family, path=path, url=url,
                                          sha256=sha(body), bytes=len(body),
                                          role="license_notice" if path in spec["license_paths"] else "source"))
            if path in spec["license_paths"]:
                continue
            text = body.decode("utf-8")
            parser = rst_paragraphs if path.endswith(".rst") else sgml_paragraphs if path.endswith(".sgml") else md_paragraphs
            passages = []
            for index, paragraph in enumerate(parser(text)):
                for part in segments(paragraph):
                    identity = sha((family + "\n" + path + "\n" + part).encode())
                    passages.append(dict(id=identity, family=family, split=spec["split"],
                                         doc_path=path, source_url=url, source_sha256=sha(body),
                                         passage_index=index, text=part, mode=mode(part)))
            passages.sort(key=lambda row: row["id"])
            selected += passages[:spec["per_file"]]
    ids = set()
    deduped = []
    for row in selected:
        normalized = re.sub(r"\s+", " ", row["text"].lower())
        if normalized in ids:
            continue
        ids.add(normalized)
        deduped.append(row)
    deduped.sort(key=lambda row: (row["split"], row["family"], row["id"]))
    for split, name in (("candidate_train", "candidate_sources.jsonl"),
                        ("untouched_holdout", "untouched_holdout.jsonl")):
        with (OUT / name).open("w", encoding="utf-8") as f:
            for row in deduped:
                if row["split"] == split:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {family: sum(row["family"] == family for row in deduped) for family in FAMILIES}
    manifest["selected_passages"] = counts
    (OUT / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"selected_passages": counts, "total": len(deduped)}))


if __name__ == "__main__":
    main()
