"""Frozen source corpus, bounded passages, and exact source-field citations."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def header(text):
    match = re.match(r"---[^\n]*\n.*?\n---[^\n]*(?:\n|$)", text, re.S)
    return (0, match.end()) if match else None


def metadata(text):
    span = header(text)
    if not span:
        return {}
    content = text[span[0]:span[1]]
    try:
        import yaml
        result = yaml.safe_load(content.split("\n", 1)[1].rsplit("\n---", 1)[0])
        return json.loads(json.dumps(result, default=str)) if isinstance(result, dict) else {}
    except Exception:
        return {k.strip(): v.strip() for line in content.splitlines() if ":" in line
                for k, v in [line.split(":", 1)]}


def sections(text):
    spans = [(m.start(), m.end(), len(m.group(1))) for m in re.finditer(r"^(#{1,6}) .+$", text, re.M)]
    for m in re.finditer(r"^([^\n]+)\n([=~^\-])\2{2,}[ \t]*(?=\n|$)", text, re.M):
        spans.append((m.start(), m.end(), 1 if m.group(2) == "=" else 2))
    fenced, opening = [], None
    for m in re.finditer(r"^[ \t]*(```|~~~).*?(?=\n|$)", text, re.M):
        if opening and opening[2] == m.group(1):
            fenced.append((opening[0], m.end()))
            opening = None
        elif not opening:
            opening = (m.start(), m.end(), m.group(1))
    if opening:
        fenced.append((opening[0], len(text)))
    return sorted((a,b,level) for a,b,level in spans if not any(x <= a < y for x,y in fenced))


def ranges(text, target=2200, maximum=3200):
    """Cover every original character; prefer paragraph/line boundaries."""
    start = 0
    while start < len(text):
        if len(text) - start <= maximum:
            yield start, len(text)
            break
        lower, upper = start + target // 2, min(len(text), start + maximum)
        preferred = min(start + target, upper)
        candidates = [m.end() for m in re.finditer(r"\n[ \t]*\n", text[start:upper])
                      if start + m.end() >= lower]
        if candidates:
            end = start + min(candidates, key=lambda p: abs(start + p - preferred))
        else:
            end = text.rfind("\n", lower, upper)
            if end < lower:
                end = text.rfind(" ", lower, upper)
            end = end + 1 if end >= lower else upper
        if end <= start:
            raise ValueError("Chunker made no progress")
        yield start, end
        start = end


def context_spans(text, start, end):
    spans = []
    h = header(text)
    if h and h[1] <= start:
        spans.append(h)
    ancestors = {}
    for a, b, level in sections(text):
        if a >= start:
            break
        ancestors = {k: v for k, v in ancestors.items() if k < level}
        ancestors[level] = (a, b)
    spans.extend(ancestors.values())
    # A short previous paragraph helps resolve local referents without inventing
    # context; it remains a separate source span, never a fabricated quote join.
    if start:
        prior = text.rfind("\n\n", max(0, start - 1100), max(0, start - 2))
        if prior >= 0:
            spans.append((prior + 2, start))
    # Preserve column labels for hard splits within a Markdown table.
    lines = list(re.finditer(r"^.*(?:\n|$)", text[:start], re.M))
    if start < len(text) and text[start:].lstrip().startswith("|"):
        for i in range(len(lines) - 1, 0, -1):
            line = lines[i].group().strip()
            if re.fullmatch(r"\|?[ :\-]+(?:\|[ :\-]+)+\|?", line):
                spans.extend([(lines[i - 1].start(), lines[i - 1].end()), (lines[i].start(), lines[i].end())])
                break
            if line and not line.startswith("|"):
                break
    # Carry the opening fence into a split code block.
    fence = None
    for m in re.finditer(r"^[ \t]*(```|~~~).*?(?=\n|$)", text[:start], re.M):
        if fence and fence[2] == m.group(1):
            fence = None
        elif not fence:
            fence = (m.start(), m.end(), m.group(1))
    if fence:
        spans.append(fence[:2])
    return sorted(set((a, min(b, start)) for a, b in spans if a < min(b, start)))


def passage(unit_key, unit, text, start, end, depth=0, parent=None):
    spans = context_spans(text, start, end)
    fragments = [text[a:b] for a, b in spans]
    if unit.get("source_context", {}).get("source_kind") == "external_issue_or_comment":
        fragments.insert(0, "Original JSON provenance (context only): " + json.dumps(unit["source_context"], ensure_ascii=False))
    target = text[start:end]
    supplied = "<source_context>\n" + "\n\n".join(fragments) + "\n</source_context>\n<target_passage>\n" + target + "\n</target_passage>"
    return {"id": "P-" + digest([unit_key, unit["sha256"], start, end])[:24], "unit_key": unit_key,
            "source_start": start, "source_end": end, "depth": depth, "parent": parent,
            "context_spans": spans, "allowed_spans": [*spans, (start, end)], "text": supplied}


def divide(case, unit, text):
    a, b = case["source_start"], case["source_end"]
    if b - a <= 300 or case["depth"] >= 3:
        return []
    middle = (a + b) // 2
    candidates = [m.end() + a for m in re.finditer(r"\n", text[a:b]) if a + (b-a)//4 < a + m.end() < b - (b-a)//4]
    split = min(candidates, key=lambda p: abs(p-middle)) if candidates else middle
    return [passage(case["unit_key"], unit, text, x, y, case["depth"] + 1, case["id"])
            for x, y in [(a, split), (split, b)]]


def prepare(root, out):
    files = sorted(subprocess.check_output(["rg", "--files"], cwd=root, text=True).splitlines())
    selected = [p for p in files if Path(p).suffix.lower() in (".md", ".txt", ".rst")
                or (p.startswith("sources/snapshots/") and p.endswith(".json"))]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Corpus already frozen; reuse it or select a new output directory")
    (out / "snapshots").mkdir(exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    sources, units, passages = {}, {}, []
    for relative in selected:
        raw = (root / relative).read_bytes()
        decoded = raw.decode("utf-8")
        key = "F-" + digest(relative)[:20]
        snapshot = "snapshots/" + key + "-" + digest(raw)[:12] + Path(relative).suffix
        (out / snapshot).write_bytes(raw)
        role = relative.split("/", 1)[0] if "/" in relative else "repository_documentation"
        sources[key] = {"path": relative, "snapshot": snapshot, "sha256": digest(raw), "bytes": len(raw), "layer": role}
        if relative.endswith(".json"):
            obj = json.loads(decoded)
            objects = [("", obj)] if isinstance(obj, dict) else [(f"/{i}", x) for i, x in enumerate(obj)]
            views = [(prefix + "/body", x["body"], {"author": (x.get("user") or {}).get("login"),
                      "url": x.get("html_url"), "created_at": x.get("created_at"), "title": x.get("title"),
                      "source_kind": "external_issue_or_comment", "report_not_independently_verified": True})
                     for prefix, x in objects if isinstance(x, dict) and isinstance(x.get("body"), str)]
        else:
            views = [(None, decoded, metadata(decoded))]
        for pointer, text, context in views:
            unit_key = "U-" + digest([relative, pointer])[:20]
            filename = "units/" + unit_key + ".txt"
            (out / filename).write_bytes(text.encode())
            unit = {"source_key": key, "text_path": filename, "sha256": digest(text),
                    "json_pointer": pointer, "source_context": context}
            units[unit_key] = unit
            for a, b in ranges(text):
                if text[a:b].strip():
                    passages.append(passage(unit_key, unit, text, a, b))
    manifest = {"version": "xdna2-bulk-v1", "root": str(root),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
                "selection": "All rg-visible Markdown, text and RST documents plus source JSON issue/comment body fields. Code, environment/config directories and generated/ignored run artifacts excluded.",
                "sources": sources, "units": units, "passages": passages,
                "semantic_support_verified": False, "upstream_references_verified": False}
    write_json(out / "manifest.json", manifest)
    return manifest


def occurrences(fragment, text, allowed):
    return [{"char_start": m.start(), "char_end": m.end(),
             "line_start": text.count("\n", 0, m.start()) + 1,
             "line_end": text.count("\n", 0, max(m.start(), m.end()-1)) + 1}
            for m in re.finditer(re.escape(fragment), text)
            if any(a <= m.start() and m.end() <= b for a, b in allowed)]


def cite(claim, case, unit, source, text, run_key):
    quotes = occurrences(claim["quote"], text, case["allowed_spans"])
    conditions = [{"text": c, "spans": occurrences(c, text, case["allowed_spans"])} for c in claim["conditions"]]
    errors = []
    if not quotes or claim["quote"] not in case["text"]:
        errors.append("Quote has no allowed exact source span")
    if any(not c["spans"] or c["text"] not in case["text"] for c in conditions):
        errors.append("One or more conditions have no allowed exact source span")
    evidence = "E-" + digest([source["sha256"], unit["json_pointer"], quotes, claim["quote"]])[:24]
    identifier = "C-" + digest([case["unit_key"], evidence, claim["statement"], claim["conditions"]])[:24]
    return {"id": identifier, "evidence_id": evidence, **claim, "source_path": source["path"],
            "source_key": unit["source_key"], "unit_key": case["unit_key"], "json_pointer": unit["json_pointer"],
            "source_sha256": source["sha256"], "unit_sha256": unit["sha256"],
            "source_snapshot": source["snapshot"], "source_spans": quotes, "condition_evidence": conditions,
            "source_layer": source["layer"], "source_context": unit["source_context"], "passage_id": case["id"],
            "citation_coordinate_system": "decoded JSON body field" if unit["json_pointer"] else "original UTF-8 document characters",
            "review_status": "proposed", "semantic_support_verified": False, "upstream_references_verified": False,
            "citation_valid": not errors, "validation_errors": errors, "run_key": run_key}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    m = prepare(a.root.resolve(), a.out.resolve())
    print(json.dumps({"files": len(m["sources"]), "document_units": len(m["units"]), "initial_passages": len(m["passages"])}))
