"""Strip filler words ("Request", "Inquiry") from training titles: data/v3 -> data/v4 (2026-10-18).

Rows whose title falls under 3 words after stripping are dropped.
"""

import json
import re
from collections import Counter
from pathlib import Path

from common.validation import MIN_WORDS, clean_title

TRAILING = re.compile(r"(?:\s+(?:and|or|for|to|of|on|about|with))*\s+(?:requests?|inquir(?:y|ies))$", re.I)
LEADING = re.compile(r"^(?:requests?|inquir(?:y|ies))\s+(?:for|to|about|on|into|regarding)\s+(?:an?\s+|the\s+)?", re.I)


def strip_filler(title: str) -> str:
    out = title
    for _ in range(2):
        out = TRAILING.sub("", out)
        out = LEADING.sub("", out)
    return clean_title(out, clamp=False)


def main() -> None:
    stats = Counter()
    examples = []
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in Path(f"data/v3/{split}.jsonl").open(encoding="utf-8") if line.strip()]
        kept = []
        for r in rows:
            new = strip_filler(r["title"])
            if new != r["title"]:
                if len(new.split()) < MIN_WORDS:
                    stats[f"{split} dropped (too short)"] += 1
                    continue
                stats[f"{split} rewritten"] += 1
                if len(examples) < 12 and stats[f"{split} rewritten"] % 997 == 1:
                    examples.append((r["title"], new))
                r["title"] = new
            kept.append(r)
        Path("data/v4").mkdir(exist_ok=True)
        Path(f"data/v4/{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        stats[f"{split} rows"] = len(kept)
    for k, v in sorted(stats.items()):
        print(f"{k}: {v:,}")
    for old, new in examples:
        print(f"  {old}  ->  {new}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
