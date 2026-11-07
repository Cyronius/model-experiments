"""Score an OpenRouter model as the meaning-check judge, many checks a request.

Like judge_eval.py (same labeled rewrites, same v4 MEANING CHECK message, the
ste_context_v5 judge prompt), but the checks go in batches through openrouter_v5. A
check missing from a reply, or not a list, counts as a flag, as an unparsed check does
in the pipeline. Results: data/out/doc-judge-eval/<name>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# data/doc_pilot/queue.py shadows the standard library's queue, which the thread pool needs.
_path, sys.path = sys.path, [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import queue  # noqa: E402,F401
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
sys.path = _path

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

sys.path[:0] = [str(Path(__file__).parent)]
from openrouter_v5 import JUDGE_BATCH, ask_batch  # noqa: E402
from ste_context_v4 import judge_message_v4  # noqa: E402
from ste_context_v5 import judge_prompt  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "data/out/doc-judge-eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", default="off")
    parser.add_argument("--variant", default="traps")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--min-n", type=int, default=0)
    args = parser.parse_args()
    lex = Lexicon()
    system = judge_prompt(lex, args.variant) + "\n\n" + JUDGE_BATCH
    pairs = [p for p in map(json.loads, (BASE / "labeled.jsonl").read_text().splitlines()) if p["n"] >= args.min_n]
    for k, p in enumerate(pairs):
        p["key"] = f"c{k}"
    chunks = [pairs[i:i + args.batch] for i in range(0, len(pairs), args.batch)]

    def run(chunk):
        blocks = [(p["key"], judge_message_v4(p["source"], p["target"], p["mode"], lex, proper_names(p["source"])))
                  for p in chunk]
        try:
            entries, usage, seconds = ask_batch(args.model, system, blocks, "checks", args.reasoning)
        except RuntimeError as exc:
            entries, usage, seconds = {}, {"error": str(exc)}, 0.0
        return [(p, entries.get(p["key"])) for p in chunk], usage, seconds

    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(run, chunks))
    rows, cost, tokens = [], 0.0, [0, 0, 0]
    for pairs_out, usage, seconds in results:
        cost += usage.get("cost") or 0
        tokens[0] += usage.get("prompt_tokens") or 0
        tokens[1] += usage.get("completion_tokens") or 0
        tokens[2] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        for p, entry in pairs_out:
            d = entry.get("differences") if entry else None
            rows.append({**p, "differences": d if isinstance(d, list) else None, "batch_seconds": round(seconds, 1)})
    flag = lambda r: r["differences"] is None or bool(r["differences"])
    changed = [r for r in rows if r["label"] == "repair"]
    kept = [r for r in rows if r["label"] == "kept"]
    out = BASE / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "judgements.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"name": args.name, "model": args.model, "variant": args.variant, "reasoning": args.reasoning,
               "batch": args.batch, "pairs": len(rows), "wall_s": round(time.time() - t0, 1),
               "meaning_changes_flagged": sum(map(flag, changed)), "of_meaning_changes": len(changed),
               "good_rewrites_flagged": sum(map(flag, kept)), "of_good_rewrites": len(kept),
               "unparsed": sum(r["differences"] is None for r in rows), "cost_usd": round(cost, 4),
               "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "reasoning_tokens": tokens[2],
               "missed_n": sorted(r["n"] for r in changed if not flag(r))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
