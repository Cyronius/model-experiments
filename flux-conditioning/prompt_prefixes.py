"""Count append-form enrichment groups in a prompt corpus: shared comma prefix -> distinct tails (2026-11-11)."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

TAGGY = re.compile(r"\b(by [a-z]+ [a-z]+|artstation|unreal engine|octane|8k|4k|highly detailed|greg rutkowski|"
                   r"trending|cgsociety|concept art|sharp focus|hd|hdr|masterpiece)\b")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" ,")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--column", default="prompt")
    parser.add_argument("--max-prefix-words", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = {norm(p) for p in pq.read_table(args.parquet, columns=[args.column]).column(0).to_pylist() if p}
    prompts.discard("")
    tails = defaultdict(set)
    segs_hist = Counter()
    for p in prompts:
        segs = [s.strip() for s in p.split(",") if s.strip()]
        segs_hist[min(len(segs), 20)] += 1
        for i in range(1, len(segs)):
            prefix = ", ".join(segs[:i])
            if len(prefix.split()) > args.max_prefix_words:
                break
            tails[prefix].add(", ".join(segs[i:]))
    sizes = Counter()
    for t in tails.values():
        n = len(t)
        sizes["2+" if n >= 2 else "1"] += 1
        for k in (5, 10, 20, 50, 100):
            sizes[f"{k}+"] += n >= k
    good = {p: t for p, t in tails.items() if len(t) >= 5}
    pairs_in_good = sum(len(t) for t in good.values())
    first_seg_words = Counter(min(len(p.split()), 12) for p in good)
    taggy_tail_share = sum(bool(TAGGY.search(tail)) for t in good.values() for tail in t) / max(pairs_in_good, 1)
    top = sorted(good.items(), key=lambda kv: -len(kv[1]))
    report = {
        "unique_prompts": len(prompts),
        "comma_segments_histogram": dict(sorted(segs_hist.items())),
        "prefixes": len(tails),
        "prefixes_by_distinct_tails": dict(sizes),
        "pairs_in_groups_with_5plus_tails": pairs_in_good,
        "prefix_word_count_for_5plus_groups": dict(sorted(first_seg_words.items())),
        "share_of_tails_with_sd_tags": round(taggy_tail_share, 3),
        "top_prefixes": [[p, len(t)] for p, t in top[:40]],
        "random_midsize_examples": [
            {"prefix": p, "n": len(t), "tails": sorted(t)[:5]}
            for p, t in [kv for kv in top if 5 <= len(kv[1]) <= 15][::5000][:12]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("top_prefixes", "random_midsize_examples")}, indent=1))


if __name__ == "__main__":
    main()
