"""Pick the new prompts to title and split them across two pods (2026-10-18).

All normal / very_short / vague / log_paste prompts, plus a capped random share of code and long prompts.
"""

import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("strat", "scripts/03_stratify.py")
strat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strat)

CAP = {"code_paste": 20_000, "long": 20_000}
by_bucket: dict[str, list[dict]] = {}
for line in Path("data/22_new_all.jsonl").open(encoding="utf-8"):
    rec = json.loads(line)
    rec["bucket"] = strat.bucket(rec["prompt"])
    by_bucket.setdefault(rec["bucket"], []).append(rec)

rng = random.Random(21)
picked = []
for name, recs in by_bucket.items():
    rng.shuffle(recs)
    picked += recs[: CAP.get(name, len(recs))]
rng.shuffle(picked)
half = len(picked) // 2
for n, part in enumerate((picked[:half], picked[half:]), start=1):
    with Path(f"data/23_title_input_{n}.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
print(len(picked), dict(Counter(r["bucket"] for r in picked)))
