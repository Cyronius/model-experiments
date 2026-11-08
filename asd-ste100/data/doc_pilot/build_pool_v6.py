"""Passage pool v6: four new documentation trees, with procedure steps, for a local teacher run.

v5's pool (build_pool_v5.py) gave 178 procedures among 12,763 accepted pairs: the v1 segmenter drops list
items and keeps sentences of 12-65 words, so numbered steps never reach it. v6 reads four more projects at
pinned commits and keeps, besides v5-style prose passages, list items:

- freebsd_handbook: freebsd/freebsd-doc documentation/content/en/books/handbook/**/*.adoc
  (FreeBSD Documentation License, BSD-2-Clause style)
- kubernetes_tasks: kubernetes/website content/en/docs/tasks/**/*.md (CC BY 4.0)
- django_docs: django/django docs/{howto,intro,topics}/**/*.txt (reStructuredText; BSD-3-Clause)
- cargo_book: rust-lang/cargo doc/book/src/**/*.md (CHANGELOG.md left out) (MIT OR Apache-2.0). The Rust Book stays the untouched
  holdout and is not read.

Prose passages: v1 parsers and segmenter, 15-55 words, self-contained, as v5. List items: the marker is
removed and a missing final period added; an item that starts with an imperative verb is a procedure
(PROC) at 6-55 words, any other item is DESC at 15-55 words. A passage is a teacher input when the checker
flags it (pool.jsonl, v5's input schema) and an identity candidate when the checker passes it
(clean.jsonl). Text already in the v5 pool or a pilot is left out. Writes data/out/doc-pool-v6/
{pool.jsonl, clean.jsonl, files.json, summary.json}. The tarballs are scratch and are deleted.
Run with data/out/.venv/bin/python.
"""

from __future__ import annotations

import html
import io
import json
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from select_sources_v4 import PARSERS, V1, example_ids, mode, normalized, segments, self_contained, sha  # noqa: E402
from select_sources_v4 import prepare  # noqa: E402
from checker import Lexicon, check, errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-pool-v6"
FAMILIES = {
    "freebsd_handbook": dict(repo="freebsd/freebsd-doc", commit="c13dc75acf41e23009b6805ba51c346091adf089",
                             prefixes=("documentation/content/en/books/handbook/",), suffix=".adoc",
                             license="FreeBSD Documentation License (BSD-2-Clause style)"),
    "kubernetes_tasks": dict(repo="kubernetes/website", commit="a79720f70372d6ff3c043847979cda5d1f01c938",
                             prefixes=("content/en/docs/tasks/",), suffix=".md", license="CC BY 4.0"),
    "django_docs": dict(repo="django/django", commit="a1f2b45ce5a327f42437bdb620a817839d0bd067",
                        prefixes=("docs/howto/", "docs/intro/", "docs/topics/"), suffix=".txt",
                        license="BSD-3-Clause"),
    "cargo_book": dict(repo="rust-lang/cargo", commit="25a58ceed46fe4754f3ad712e6c98ff9cd669317",
                       prefixes=("doc/book/src/",), suffix=".md", skip=("doc/book/src/CHANGELOG.md",), license="MIT OR Apache-2.0"),
}
IMPERATIVE = re.compile(
    r"^(?:Use|Run|Set|Copy|Open|Close|Write|Read|Check|Start|Stop|Do|Make|Call|Enter|Select|Specify|Add|Remove|"
    r"Create|Delete|Put|Keep|Send|Save|Press|Install|Edit|Type|Click|Choose|Configure|Verify|Ensure|Download|"
    r"Apply|Change|Replace|Load|Mount|Unmount|Log|Navigate|Go|Follow|Restart|Reboot|Update|Upgrade|Define|Insert|"
    r"Boot|Build|Deploy|Enable|Disable|Attach|Detach|Connect|Disconnect|Wait|Find|Look|Get|Give|Move|Rename|"
    r"Test|Look|Point|Place|Pass|Expose|Scale|Fetch|Publish|Register|Configure|Uncomment|Comment|Allow|Do not|Don't|"
    r"Make sure|Repeat|Return|Restore|Back up|Plug|Unplug|Insert|Format|Partition|Compile|Commit|Push|Pull|Clone|"
    r"Generate|Append|Prepare|Identify|Review|Examine|Inspect|Modify|Adjust|Assign|Bind|Grant|Label|Annotate|Drain|"
    r"Cordon|Patch|Migrate|Convert|Extract|Unpack|Provide|Include|Exclude|Skip|Upload|Import|Export|Sign|Encrypt|"
    r"Decrypt|Initialize|Initialise|Execute|Invoke|Launch|Shut|Power|Turn|Switch|Clear|Reset|Flush|Kill|Terminate|"
    r"Tag|Specify|Declare|Store|Write|Paste|Visit|Browse|Search|Refer|Consult|Contact|Report|Mark|Lock|Unlock|"
    r"Uninstall|Reinstall|Rebuild|Recompile|Re-run|Rerun|Retry|Confirm|Accept|Reject|Approve|Deny|Answer|Specify)\b")
# "Build dependencies will ...", "Pull requests are ...": a noun phrase, not a command
NOT_COMMAND = re.compile(r"^\S+\s+(?:\S+\s+)?(?:is|are|was|were|will|can|could|may|might|must|should|would|has|have|does)\b")
LIST_MARKER = re.compile(r"^\s*(?:[*+-]|\d+[.)]|#\.|\.{1,3}|\*{1,3})\s+(?=\S)")


def tarball(repo: str, commit: str) -> bytes:
    req = urllib.request.Request(f"https://codeload.github.com/{repo}/tar.gz/{commit}",
                                 headers={"User-Agent": "asd-ste100-doc-pool-v6"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def adoc_inline(value: str) -> str:
    value = re.sub(r"\[\.[\w-]+\]#([^#]+)#", r"`\1`", value)
    value = re.sub(r"(?<!\w)#([^#\s][^#]*)#(?!\w)", r"\1", value)
    value = value.replace("(R)", "").replace("(TM)", "")
    value = re.sub(r"(?:link|xref|https?://\S+?)(?::\S*?)?\[([^\]]*)\]", lambda m: m[1], value)
    value = re.sub(r"<<[^,>]+,([^>]+)>>", r"\1", value)
    value = re.sub(r"<<([^>]+)>>", r"\1", value)
    value = re.sub(r"man:([\w.+-]+)\[(\w+)\]", r"`\1(\2)`", value)
    value = re.sub(r"footnote:\[[^\]]*\]", "", value)
    value = re.sub(r"`\+?([^`+]+)\+?`", r"`\1`", value)
    value = re.sub(r"(?<!\w)[*_]{1,2}([^*_]+?)[*_]{1,2}(?!\w)", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([,;:.!?])", r"\1", value)


def adoc_blocks(raw: str):
    """Paragraph blocks of an AsciiDoc file, outside listings, tables, comments and attribute lines."""
    out, block, fence = [], [], None
    for line in raw.splitlines():
        s = line.strip()
        if fence:
            if s == fence:
                fence = None
            continue
        if re.fullmatch(r"(-{4,}|\.{4,}|/{4,}|\+{4,}|\|={3,})", s):
            fence = s
            continue
        if not s or s == "====" or s.startswith(("[", "//", "=", ":", "include::", "image::", "ifdef::", "endif::",
                                                 "ifndef::", "toc::")):
            if block:
                out.append(block)
            block = []
            continue
        block.append(s)
    if block:
        out.append(block)
    return out


def split_items(lines: list[str]):
    """(text, is_item) for one block: list items one by one, other text joined."""
    if not LIST_MARKER.match(lines[0]):
        yield " ".join(lines), False
        return
    item = []
    for line in lines:
        if LIST_MARKER.match(line):
            if item:
                yield " ".join(item), True
            item = [LIST_MARKER.sub("", line, count=1)]
        else:
            item.append(line.strip())
    if item:
        yield " ".join(item), True


def blocks(raw: str, suffix: str):
    """(clean text, is_item) for a file."""
    if suffix == ".adoc":
        for lines in adoc_blocks(raw):
            for text, item in split_items(lines):
                if "{" not in text and "::" not in text:
                    yield adoc_inline(html.unescape(text)), item
        return
    if suffix == ".md":
        raw = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.S)
        raw = re.sub(r"<!--.*?-->|```.*?```|\{\{[<%].*?[>%]\}\}", " ", raw, flags=re.S)
    for chunk in re.split(r"\n\s*\n", raw):
        lines = [l.rstrip() for l in chunk.splitlines() if l.strip()]
        if not lines:
            continue
        if LIST_MARKER.match(lines[0]) and not (suffix != ".md" and lines[0].startswith("   ")):
            for text, _ in split_items([l.strip() for l in lines]):
                if suffix == ".md":
                    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
                if not re.search(r"(?:^|\s)\.\. |::|\{\{|\}\}", text):
                    yield prepare.clean_markup(text), True
            continue
        for value in PARSERS[".md" if suffix == ".md" else ".rst"]("\n".join(lines)):
            yield value, False


def item_passage(text: str):
    """(passage, mode) for a list item, or None."""
    text = text.strip()
    if not text or any(t in text for t in ("<", ">", "\\", "&quot;", "{{", "}}")):
        return None
    if text.count("`") % 2 or text.count('"') % 2 or text.startswith("`"):
        return None
    if not text.endswith((".", "!", "?", ":")):
        text += "."
    words = len(text.split())
    if IMPERATIVE.match(text) and not NOT_COMMAND.match(text):  # a step that introduces a listing may end with a colon
        return (text, "PROC") if 6 <= words <= 55 else None
    return (text, "DESC") if 15 <= words <= 55 and not text.endswith(":") else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {normalized(json.loads(l)["text"]) for l in (ROOT / "data/out/doc-pool-v5/pool.jsonl").open()}
    for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl", "candidate_sources.jsonl"):
        seen |= {normalized(json.loads(l)["text"]) for l in (V1 / name).read_text().splitlines()}
    used_ids = example_ids()
    lex = Lexicon()
    pool, clean, files, counts = [], [], [], Counter()
    for family, spec in FAMILIES.items():
        data = tarball(spec["repo"], spec["commit"])
        counts[family + ":tarball_mb"] = round(len(data) / 2**20)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                path = member.name.split("/", 1)[1] if "/" in member.name else member.name
                if not (member.isfile() and path.startswith(spec["prefixes"]) and path.endswith(spec["suffix"])
                        and path not in spec.get("skip", ())):
                    continue
                body = tar.extractfile(member).read()
                try:
                    text = body.decode("utf-8")
                except UnicodeDecodeError:
                    counts[family + ":undecodable"] += 1
                    continue
                url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
                files.append(dict(family=family, path=path, url=url, sha256=sha(body), license=spec["license"]))
                for index, (value, is_item) in enumerate(blocks(text, spec["suffix"])):
                    if is_item:
                        got = item_passage(value)
                        parts = [got] if got else []
                    else:
                        parts = [(p, mode(p)) for p in segments(value) if 15 <= len(p.split()) <= 55]
                    for part, md in parts:
                        if not is_item and IMPERATIVE.match(part) and not NOT_COMMAND.match(part):
                            md = "PROC"
                        key = normalized(part)
                        row = dict(id=sha((family + "\n" + path + "\n" + part).encode()), family=family,
                                   split="candidate_train", doc_path=path, source_url=url, source_sha256=sha(body),
                                   passage_index=index, text=part, mode=md, list_item=is_item)
                        if row["id"] in used_ids or key in seen or not self_contained(row):
                            continue
                        seen.add(key)
                        rules = [v.rule for v in errors(check(part, lex, md))]
                        row.update(checker_error_rules=rules, word_count=len(part.split()))
                        (pool if rules else clean).append(row)
                        counts[f"{family}:{'pool' if rules else 'clean'}:{md}"] += 1
        del data
        print(json.dumps({k: v for k, v in counts.items() if k.startswith(family)}), flush=True)
    for name, rows in (("pool.jsonl", pool), ("clean.jsonl", clean)):
        rows.sort(key=lambda r: r["id"])
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (OUT / "files.json").write_text(json.dumps(files, indent=1) + "\n")
    summary = dict(schema="asd-ste100-doc-pool-v6", pool=len(pool), clean=len(clean),
                   by_family=dict(Counter(r["family"] for r in pool)),
                   by_mode=dict(Counter(r["mode"] for r in pool)),
                   clean_by_mode=dict(Counter(r["mode"] for r in clean)), files=len(files), counts=dict(counts),
                   commits={f: s["commit"] for f, s in FAMILIES.items()}, holdout="rust_book (not read)")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
