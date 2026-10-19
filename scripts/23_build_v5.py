"""Build data/v5: v4 plus the newly titled WildChat-4.8M and synthetic prompts (2026-10-18).

New rows go to train only; val and test stay exactly as in v4. New titles get the same cleaning as v4:
drop user-narration titles, strip Request/Inquiry filler, drop anything under 3 words or matching the scrub.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

from common.validation import MIN_WORDS, validate_title


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


narration = load("narr", "scripts/09_filter_boilerplate.py").NARRATION
strip_filler = load("fill", "scripts/18_strip_filler.py").strip_filler
scrub = load("scrub", "scripts/17b_scrub_minors_sexual.py")


def read(path: str):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


titles, prompts = {}, {}
for r in read("data/26_titled_all.jsonl"):
    prompts[r["id"]] = r
    if r.get("title"):
        titles[r["id"]] = r["title"]
for r in read("data/26_luna.jsonl"):
    if r.get("title") and r["id"] not in titles:
        titles[r["id"]] = r["title"]

stats, new_rows = Counter(), []
for pid, r in prompts.items():
    title = titles.get(pid)
    if not title:
        stats["no title from either teacher"] += 1
    elif narration.search(title):
        stats["dropped: narration"] += 1
    else:
        title = strip_filler(title)
        if len(title.split()) < MIN_WORDS:
            stats["dropped: under 3 words after stripping"] += 1
        elif validate_title(title, r["prompt"]):
            stats["dropped: failed validation after stripping"] += 1
        elif scrub.flagged({"prompt": r["prompt"], "title": title}):
            stats["dropped: scrub"] += 1
        else:
            new_rows.append({"id": pid, "bucket": r["bucket"], "prompt": r["prompt"], "title": title})
            stats["kept"] += 1

out = Path("data/v5")
out.mkdir(exist_ok=True)
train = read("data/v4/train.jsonl") + new_rows
for split, rows in (("train", train), ("val", read("data/v4/val.jsonl")), ("test", read("data/v4/test.jsonl"))):
    (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{split}: {len(rows):,}")
for k, v in stats.items():
    print(f"  new rows {k}: {v:,}")
print("new-row buckets:", dict(Counter(r["bucket"] for r in new_rows)))
