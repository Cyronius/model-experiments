"""A second meaning check through OpenRouter for the accepted rewrites of runs, as a review flag.

Like local_rejudge_v5.py, but the checks go in batches to an OpenRouter model (by default
gpt-oss-120b, a family apart from the rewriters and from the DeepSeek judge) and the
records go to second_judge_or.jsonl in each run directory: {"id", "judge", "differences"}.
Nothing is removed. A check missing from a reply, or not a list, is not recorded, and is
asked again in the next pass (up to --passes); a rerun skips the ids already checked.
The key is read from OPENROUTER_API_KEY.
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
import threading  # noqa: E402

sys.path[:0] = [str(Path(__file__).parent)]
from openrouter_v5 import JUDGE_BATCH, ask_batch  # noqa: E402
from ste_context_v4 import judge_message_v4  # noqa: E402
from ste_context_v5 import judge_prompt  # noqa: E402
from synthesize import append  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "data/out/doc-teacher-v5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="run directories, comma-separated")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--variant", default="traps")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=3.0, help="US dollars for this invocation")
    args = parser.parse_args()
    lex = Lexicon()
    system = judge_prompt(lex, args.variant) + "\n\n" + JUDGE_BATCH
    lock, spent = threading.Lock(), [0.0]

    def check(out, chunk):
        with lock:
            if spent[0] >= args.max_cost:
                return 0
        blocks = [(f"c{k}", judge_message_v4(r["text"], r["ste"], r["mode"], lex, proper_names(r["text"])))
                  for k, r in enumerate(chunk)]
        try:
            entries, usage, _ = ask_batch(args.model, system, blocks, "checks", args.reasoning)
        except RuntimeError:
            return 0
        done = 0
        with lock:
            spent[0] += usage.get("cost") or 0
            for k, r in enumerate(chunk):
                d = (entries.get(f"c{k}") or {}).get("differences")
                if isinstance(d, list):
                    append(out / "second_judge_or.jsonl", {"id": r["id"], "judge": args.model, "differences": d})
                    done += 1
        return done

    for run in args.runs.split(","):
        out = BASE / run
        for _ in range(args.passes):
            path = out / "second_judge_or.jsonl"
            seen = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
            rows = [r for r in map(json.loads, (out / "accepted.jsonl").read_text().splitlines()) if r["id"] not in seen]
            if not rows:
                break
            chunks = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
            with ThreadPoolExecutor(args.parallel) as pool:
                if not sum(pool.map(lambda c: check(out, c), chunks)):
                    break
        checked = [json.loads(l) for l in (out / "second_judge_or.jsonl").read_text().splitlines()] \
            if (out / "second_judge_or.jsonl").exists() else []
        accepted = sum(1 for _ in (out / "accepted.jsonl").open())
        print(json.dumps({"run": run, "accepted": accepted, "checked": len(checked),
                          "flagged": sum(1 for r in checked if r["differences"]), "spent": round(spent[0], 4)}), flush=True)


if __name__ == "__main__":
    main()
