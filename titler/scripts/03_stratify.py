"""Stage 3: bucket prompts by difficulty heuristics and sample a stratified set.

Buckets: code_paste, log_paste, very_short, vague, long, normal.
Hard buckets get a forced minimum share so the student sees them in training.

Usage:
    python scripts/03_stratify.py --input data/02_deduped.jsonl --output data/03_sampled.jsonl --total 200000
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CODE_RE = re.compile(r"```|[{};]\s*$|^\s{4,}\S|def |function |class |import |#include|=>", re.M)
LOG_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|Traceback \(most recent|at [\w.$]+\([\w.]+:\d+\)"
    r"|^\s*(ERROR|WARN|INFO|FATAL)\b|Exception\b|errno|stack trace",
    re.M | re.I,
)
VAGUE_WORDS = {"help", "hi", "hello", "hey", "please", "can", "you", "me", "with", "something", "need", "want", "a", "i", "my", "this", "it", "do", "how", "what"}

# minimum share of the sample per hard bucket (oversampled vs natural rate)
MIN_SHARE = {"code_paste": 0.08, "log_paste": 0.08, "very_short": 0.08, "vague": 0.08, "long": 0.08}


def bucket(prompt: str) -> str:
    words = prompt.split()
    if len(words) <= 3:
        return "very_short"
    if len(words) > 500:
        return "long"
    code_hits = len(CODE_RE.findall(prompt))
    if code_hits >= 2 or (code_hits >= 1 and len(words) > 30):
        return "code_paste"
    if LOG_RE.search(prompt):
        return "log_paste"
    if len(words) <= 8 and sum(w.lower() in VAGUE_WORDS for w in words) / len(words) > 0.6:
        return "vague"
    return "normal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--total", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["bucket"] = bucket(rec["prompt"])
            by_bucket[rec["bucket"]].append(rec)

    rng = random.Random(args.seed)
    for recs in by_bucket.values():
        rng.shuffle(recs)

    total = min(args.total, sum(len(v) for v in by_bucket.values()))
    picked: list[dict] = []
    # forced minimums for hard buckets first
    for name, share in MIN_SHARE.items():
        want = int(total * share)
        got = by_bucket.get(name, [])[:want]
        picked.extend(got)
        by_bucket[name] = by_bucket.get(name, [])[len(got):]
    # fill the remainder proportionally from what's left (mostly normal)
    remainder = total - len(picked)
    leftovers = [r for recs in by_bucket.values() for r in recs]
    rng.shuffle(leftovers)
    picked.extend(leftovers[:remainder])
    rng.shuffle(picked)

    counts: dict[str, int] = defaultdict(int)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for rec in picked:
            counts[rec["bucket"]] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"sampled {len(picked)} of requested {args.total}")
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n} ({n / len(picked):.1%})")


if __name__ == "__main__":
    main()
