"""Stream fresh English first-turn prompts from allenai/WildChat-4.8M (2026-10-18).

Skips rows WildChat marks toxic, redacted, or moderation-flagged. Dedup against the existing pools is the
next step (scripts/02_clean_dedup.py on old + new, then keep the new ids).

    python scripts/20_pull_wildchat48.py --output data/20_wc48_raw.jsonl --max-rows 2600000 --max-kept 700000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=2_600_000)
    ap.add_argument("--max-kept", type=int, default=700_000)
    args = ap.parse_args()

    ds = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    kept = 0
    with Path(args.output).open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= args.max_rows or kept >= args.max_kept:
                break
            if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
                continue
            conv = row.get("conversation") or []
            if not conv or conv[0].get("role") != "user" or conv[0].get("toxic") or conv[0].get("redacted"):
                continue
            mod = row.get("openai_moderation") or []
            if mod and mod[0].get("flagged"):
                continue
            text = (conv[0].get("content") or "").strip()
            if not text:
                continue
            f.write(json.dumps({"id": f"w48_{i}", "source": "wildchat48", "prompt": text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50_000 == 0:
                print(f"{kept:,} kept after {i + 1:,} rows", flush=True)
    print(f"done: {kept:,} kept -> {args.output}")


if __name__ == "__main__":
    main()
