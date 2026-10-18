"""Write a FineWeb-Edu text slice for the LM stage (2026-10-17). Runs on the training pod.

    python scripts/14_fetch_fineweb.py --out data/fineweb_edu.jsonl --chars 1.2e9
"""

import argparse
import json

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", type=float, default=1.2e9)
    ap.add_argument("--doc-chars", type=int, default=6000)
    args = ap.parse_args()
    total = 0
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in stream:
            text = row["text"][: args.doc_chars]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total += len(text)
            if total >= args.chars:
                break
    print(f"{total / 1e6:.0f}M chars -> {args.out}")


if __name__ == "__main__":
    main()
