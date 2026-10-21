# How often and how early each shrunk model's greedy answer departs from the stock 2B's, plus judge batches for Haiku (2026-10-19).
import difflib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
prompts = {p["id"]: p for p in map(json.loads, (HERE / "prompts.jsonl").open(encoding="utf-8"))}


def load(name):
    return {a["id"]: a for a in map(json.loads, (HERE / f"answers_{name}.jsonl").open(encoding="utf-8"))}


ref = load("stock")
models = sys.argv[1:] or ["mlp75", "mlp50-selfchat"]
summary = {}
for m in models:
    ans = load(m)
    rows = []
    for i, p in prompts.items():
        a, b = ref[i]["answer"], ans[i]["answer"]
        prefix = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rows.append({"id": i, "cat": p["cat"], "identical": a == b, "shared_prefix_chars": prefix,
                     "prefix_frac": prefix / max(1, len(a)), "similarity": difflib.SequenceMatcher(None, a, b).ratio(),
                     "hit_limit": ans[i]["finish"] == "length", "ref_hit_limit": ref[i]["finish"] == "length"})
    n = len(rows)
    summary[m] = {
        "identical": sum(r["identical"] for r in rows) / n,
        "median_shared_prefix_chars": sorted(r["shared_prefix_chars"] for r in rows)[n // 2],
        "mean_similarity": sum(r["similarity"] for r in rows) / n,
        "hit_token_limit": sum(r["hit_limit"] for r in rows) / n,
        "stock_hit_token_limit": sum(r["ref_hit_limit"] for r in rows) / n,
        "rows": rows,
    }
    print(f"{m}: identical {summary[m]['identical']:.0%}  median shared prefix {summary[m]['median_shared_prefix_chars']} chars  "
          f"similarity {summary[m]['mean_similarity']:.2f}  hit 400-token limit {summary[m]['hit_token_limit']:.0%} "
          f"(stock {summary[m]['stock_hit_token_limit']:.0%})")
(HERE / "divergence.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
