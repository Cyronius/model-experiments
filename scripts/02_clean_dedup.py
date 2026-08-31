"""Stage 2: clean, length-filter, exact-dedup, and MinHash near-dedup prompts.

Usage:
    python scripts/02_clean_dedup.py --input data/01_raw_prompts.jsonl --output data/02_deduped.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

MIN_CHARS, MAX_CHARS = 2, 8000
NUM_PERM = 64
NEAR_DUP_THRESHOLD = 0.85


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def shingles(text: str, k: int = 5):
    tokens = text.split()
    if len(tokens) <= k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    seen_exact: set[str] = set()
    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=NUM_PERM)
    kept = dropped_len = dropped_exact = dropped_near = 0

    with Path(args.input).open(encoding="utf-8") as fin, Path(args.output).open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt = rec["prompt"].strip()
            if not (MIN_CHARS <= len(prompt) <= MAX_CHARS):
                dropped_len += 1
                continue
            norm = normalize(prompt)
            if norm in seen_exact:
                dropped_exact += 1
                continue
            seen_exact.add(norm)

            # near-dedup only matters for multi-word prompts; short ones are
            # already caught by the exact check
            if len(norm.split()) >= 10:
                mh = MinHash(num_perm=NUM_PERM)
                for sh in shingles(norm):
                    mh.update(sh.encode("utf-8"))
                if lsh.query(mh):
                    dropped_near += 1
                    continue
                lsh.insert(rec["id"], mh)

            rec["prompt"] = prompt
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"kept {kept} | dropped: length {dropped_len}, exact-dup {dropped_exact}, near-dup {dropped_near}"
    )


if __name__ == "__main__":
    main()
