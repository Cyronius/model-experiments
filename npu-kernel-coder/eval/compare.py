"""Paired comparison of two runs on the tasks both completed.

    python3 compare.py runs/gpt-6.1-sol-medium runs/gpt-6.1-sol-medium-ex1
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(d: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in d.glob("*/result.json")}


def main() -> None:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(a_dir), load(b_dir)
    ids = sorted(set(a) & set(b))
    print(f"{len(ids)} tasks in both: {a_dir.name} vs {b_dir.name}\n")
    print(f"| difficulty | n | exact {a_dir.name} | exact {b_dir.name} | gap median | gap median | gap better / same / worse |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for diff in ("easy", "medium", "hard", "all"):
        sel = [i for i in ids if diff == "all" or a[i]["difficulty"] == diff]
        if not sel:
            continue
        ga = [a[i]["score"]["gap_closed"] for i in sel]
        gb = [b[i]["score"]["gap_closed"] for i in sel]
        better = sum(y > x + 1e-9 for x, y in zip(ga, gb))
        worse = sum(y < x - 1e-9 for x, y in zip(ga, gb))
        print(f"| {diff} | {len(sel)} | {sum(a[i]['score']['exact'] for i in sel)} | {sum(b[i]['score']['exact'] for i in sel)} | "
              f"{statistics.median(ga):.3f} | {statistics.median(gb):.3f} | {better} / {len(sel) - better - worse} / {worse} |")
    flips = [(i, a[i]["score"]["exact"], b[i]["score"]["exact"]) for i in ids if a[i]["score"]["exact"] != b[i]["score"]["exact"]]
    if flips:
        print("\nexact changed:", ", ".join(f"{i} {'x' if x else '-'}->{'x' if y else '-'}" for i, x, y in flips))
    for tag, r in ((a_dir.name, a), (b_dir.name, b)):
        sel = [r[i] for i in ids]
        print(f"{tag}: builds {sum(x['score']['builds'] for x in sel)}/{len(sel)}, rule gate {sum(x['score']['rules_pass'] for x in sel)}/{len(sel)}, "
              f"bytes {sum(x['score']['bytes_match'] for x in sel)}/{len(sel)}")


if __name__ == "__main__":
    main()
