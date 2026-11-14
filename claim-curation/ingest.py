#!/usr/bin/env python3
"""Snapshot new UTF-8 documents into the same repeatable curation pipeline.

Paragraph/fence-aware chunk boundaries are a soft size target, not semantic
guarantees. Oversize blocks are retained and marked, never silently truncated.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks(text, target=3000):
    start, offset, in_fence = 0, 0, False
    fence = ""
    for line in text.splitlines(keepends=True):
        token = line.lstrip()
        if token.startswith(("```", "~~~")):
            kind = token[:3]
            if not in_fence:
                in_fence, fence = True, kind
            elif kind == fence:
                in_fence = False
        offset += len(line)
        if offset - start >= target and not line.strip() and not in_fence:
            yield start, offset
            start = offset
    if offset > start:
        yield start, offset


def ingest(root, paths, out, target=3000):
    root = root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    sources, passages = {}, []
    for name in paths:
        path = (root / name).resolve()
        relative = path.relative_to(root).as_posix()
        text = path.read_bytes().decode("utf-8")
        key = "S-" + sha(relative)[:16]
        content_hash = sha(text)
        snapshot = f"sources/{key}-{content_hash[:12]}.txt"
        (out / snapshot).write_bytes(text.encode("utf-8"))
        header = ""
        if text.startswith("---"):
            match = re.match(r"---[^\n]*\n(.*?)\n---[^\n]*(?:\n|$)", text, re.S)
            header = match.group(0) if match else ""
        context = {}
        for line in header.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                context[k.strip()] = v.strip()
        sources[key] = {"path": relative, "snapshot": snapshot, "sha256": content_hash,
                        "context": context, "source_level": "unreviewed_input_document",
                        "upstream_reference_text": next((l for l in text.splitlines() if l.startswith("Sources")), ""),
                        "upstream_references_verified": False}
        for start, end in chunks(text, target):
            excerpt = text[start:end]
            if not excerpt.strip():
                continue
            identifier = "P-" + sha(json.dumps([relative, content_hash, start, end]))[:20]
            # Repeat document-level metadata for later chunks, without inventing
            # source spans across the gap between header and excerpt.
            payload = excerpt if start == 0 or not header else header + "\n" + excerpt
            passages.append({"id": identifier, "source_key": key, "synthetic": False,
                             "text": payload, "excerpt": excerpt, "source_start": start,
                             "source_end": end, "oversize_block": len(excerpt) > target * 2})
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    manifest = {"version": "ingest-v1", "notes_commit": revision,
                "annotation": "Unannotated inputs; do not interpret absent gold as a no-claims control.",
                "sources": sources, "passages": passages}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not (out / "gold.json").exists():
        (out / "gold.json").write_text("{}\n")
    if not (out / "pairs.json").exists():
        (out / "pairs.json").write_text("[]\n")
    print(f"Snapshotted {len(sources)} documents into {len(passages)} passages at {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--chunk-chars", type=int, default=3000)
    a = p.parse_args()
    ingest(a.root, a.paths, a.out, a.chunk_chars)
