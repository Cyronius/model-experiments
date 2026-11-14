#!/usr/bin/env python3
"""Prepare document-disjoint teacher inputs without using pilot benchmark labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(notes, baseline, out, per_train_app=2):
    old = json.loads((baseline / "manifest.json").read_text())
    excluded = {s["path"] for s in old["sources"].values()}
    split_apps = {"train": ["phlegm", "fastflowlm", "openflowlm-next"],
                  "validation": ["hipfire"], "test": ["ggml-xdna"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    sources, examples = {}, []
    hashes = {}
    for split, apps in split_apps.items():
        for app in apps:
            paths = sorted((notes / "raw" / app).glob("*.md"), key=lambda p: digest(p.name))
            selected = 0
            for path in paths:
                relative = path.relative_to(notes).as_posix()
                if relative in excluded:
                    continue
                text = path.read_bytes().decode("utf-8")
                sha = digest(text)
                if sha in hashes:
                    continue
                header_end = text.find("\n---", 3) + 4 if text.startswith("---") else 0
                header = text[:header_end]
                # Keep source blocks intact; use bounded prose/list observations,
                # not metadata, reference lists, or benchmark annotations.
                blocks = []
                for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text[header_end:], re.S):
                    block = match.group()
                    if 100 <= len(block) <= 1600 and not block.startswith(("#", "Sources", "```", "</")):
                        blocks.append((header_end + match.start(), header_end + match.end(), block))
                if len(blocks) < 2:
                    continue
                key = "S-" + digest(relative)[:16]
                filename = f"sources/{key}-{sha[:12]}.md"
                (out / filename).write_bytes(text.encode())
                sources[key] = {"path": relative, "sha256": sha, "snapshot": filename,
                                "split": split, "project": app, "original_references_verified": False}
                hashes[sha] = split
                for start, end, block in blocks[:2]:
                    # Include the closest section heading as context without
                    # manufacturing source text or inventing missing qualifiers.
                    headings = list(re.finditer(r"^#{1,6} .+$", text[:start], re.M))
                    heading = headings[-1].group() if headings else ""
                    title_match = re.search(r"^# .+$", text[:start], re.M)
                    title = title_match.group() if title_match else ""
                    context_headings = "\n\n".join(dict.fromkeys(x for x in (title, heading) if x))
                    supplied = header + "\n\n" + context_headings + "\n\n" + block
                    example_id = "D-" + digest([relative, sha, start, end])[:20]
                    examples.append({"id": example_id, "split": split, "source_key": key,
                                     "source_sha256": sha, "document": supplied,
                                     "body": block, "source_start": start, "source_end": end,
                                     "heading": heading, "title": title, "task": "extraction"})
                selected += 1
                if selected >= per_train_app:
                    break
            if selected < per_train_app:
                raise ValueError(f"Only {selected} usable documents for {app}")
    groups = {split: {x["source_key"] for x in examples if x["split"] == split} for split in split_apps}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError("Document leakage across splits")
    manifest = {"version": "distill-v2", "sources": sources, "split_projects": split_apps,
                "excluded_pilot_sources": sorted(excluded),
                "annotation": "Teacher inputs only; no teacher labels or training performed by preparation.",
                "examples": examples}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "inputs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--documents-per-project", type=int, default=2)
    a = p.parse_args()
    m = prepare(a.notes, a.baseline, a.out, a.documents_per_project)
    print(json.dumps({"documents": len(m["sources"]), "examples": len(m["examples"]),
                      "by_split": {s: sum(e["split"] == s for e in m["examples"]) for s in m["split_projects"]}}, indent=2))
