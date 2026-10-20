"""Win rate against v1.5 by input type, plus the failing rows for each type (2026-10-20).

    uv run python scripts/25_hard_input_report.py --dump data/eval/judge_rust_q4_vs_v15.jsonl --out data/eval/hard_failures.json
"""

import argparse
import collections
import json
import re


def non_english(p: str) -> bool:
    letters = [c for c in p if c.isalpha()]
    if not letters:
        return False
    if sum(1 for c in letters if ord(c) > 0x024F) / len(letters) > 0.3:
        return True
    accented = re.search(r"[¿¡ñãõçßäöüéèêàâîôûœ]", p)
    return bool(accented) and len(re.findall(r"\b(the|and|is|to|of|a|in|for|you|what)\b", p.lower())) < 2


def kinds(row: dict) -> list[str]:
    p, bucket = row["prompt"], row.get("bucket", "normal")
    out = []
    if non_english(p):
        out.append("non_english")
    if bucket in ("code_paste", "sql", "log_paste", "markdown", "long"):
        out.append("pasted_" + bucket)
    if bucket in ("very_short", "vague") or len(p.split()) <= 4:
        out.append("short_or_vague")
    return out or [bucket if bucket in ("hard", "gratitude", "hostile") else "other"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="judge dump with the new titles as A and v1.5 as B")
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_set, encoding="utf-8") if l.strip()]
    judged = {d["id"]: d for d in map(json.loads, open(args.dump, encoding="utf-8")) if d}
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    failing: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        d = judged.get(r["id"])
        verdict = d["verdict"] if d else "same"
        for k in kinds(r):
            stats[k]["rows"] += 1
            stats[k][verdict] += 1
            if verdict in ("b", "tie"):
                failing[k].append({"id": r["id"], "verdict": "v1.5 wins" if verdict == "b" else "tie", "prompt": r["prompt"][:600], "new": d["a"], "v15": d["b"]})
    print(f"{'type':16} {'rows':>5} {'same':>5} {'new':>4} {'v1.5':>5} {'tie':>4}  new share of decided")
    for k, c in sorted(stats.items(), key=lambda x: -x[1]["rows"]):
        decided = c["a"] + c["b"]
        print(f"{k:16} {c['rows']:5} {c['same']:5} {c['a']:4} {c['b']:5} {c['tie']:4}  {c['a'] / decided:.1%}" if decided else f"{k:16} {c['rows']:5} {c['same']:5}")
    json.dump(failing, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("failing rows (v1.5 wins or tie):", {k: len(v) for k, v in failing.items()}, "->", args.out)


if __name__ == "__main__":
    main()
