"""Stage 5: final validation of titled pairs and stratified train/val/test split.

Usage:
    python scripts/05_validate_split.py --input data/04_titled.jsonl --outdir data
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common.validation import validate_title

SPLITS = {"train": 0.98, "val": 0.01, "test": 0.01}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_bucket = defaultdict(list)
    dropped = 0
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("title") or validate_title(rec["title"], rec["prompt"]) is not None:
                dropped += 1
                continue
            by_bucket[rec.get("bucket", "normal")].append(
                {"id": rec["id"], "bucket": rec.get("bucket"), "prompt": rec["prompt"], "title": rec["title"]}
            )

    rng = random.Random(args.seed)
    out = {name: [] for name in SPLITS}
    for recs in by_bucket.values():
        rng.shuffle(recs)
        n = len(recs)
        n_val = max(1, int(n * SPLITS["val"])) if n >= 10 else 0
        n_test = max(1, int(n * SPLITS["test"])) if n >= 10 else 0
        out["val"].extend(recs[:n_val])
        out["test"].extend(recs[n_val : n_val + n_test])
        out["train"].extend(recs[n_val + n_test :])

    outdir = Path(args.outdir)
    for name, recs in out.items():
        rng.shuffle(recs)
        path = outdir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{name}: {len(recs)} -> {path}")
    print(f"dropped invalid: {dropped}")


if __name__ == "__main__":
    main()
