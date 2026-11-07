"""A second meaning check on this machine for the accepted rewrites of a run, as a review flag.

Reads data/out/doc-teacher-v5/<run>/accepted.jsonl, asks the model served by a local
llama-server (one slot per worker, the cached judge prefix restored before each check,
as judge_eval.py does) for the v4 MEANING CHECK of each rewrite, and appends
{"id", "differences", ...} to second_judge.jsonl in the run directory. Nothing is
removed: a rewrite that the second judge flags is a candidate for the owner's spot
checks. A rerun skips the ids already checked.
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
from ste_context_v4 import judge_message_v4  # noqa: E402
from ste_context_v5 import judge_prompt  # noqa: E402
from ste_context_v3 import parse_judge  # noqa: E402
from synthesize_v2 import post  # noqa: E402
from synthesize_v5 import SLOTS, prepare_prefix  # noqa: E402
from synthesize import append  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "data/out/doc-teacher-v5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--kwargs", default='{"enable_thinking": false}')
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--variant", default="traps")
    args = parser.parse_args()
    lex, kwargs, out = Lexicon(), json.loads(args.kwargs), BASE / args.run
    system = judge_prompt(lex, args.variant)
    SLOTS.mkdir(parents=True, exist_ok=True)
    prefix = prepare_prefix(args.url, system, {"kwargs": kwargs}, args.model_id, 7200)
    done = {json.loads(l)["id"] for l in (out / "second_judge.jsonl").read_text().splitlines()} \
        if (out / "second_judge.jsonl").exists() else set()
    rows = [r for r in map(json.loads, (out / "accepted.jsonl").read_text().splitlines()) if r["id"] not in done]
    free, lock = list(range(args.parallel)), threading.Lock()

    def check(row):
        with lock:
            slot = free.pop()
        try:
            post(args.url, f"/slots/{slot}?action=restore", {"filename": prefix["file"]}, 600)
            message = judge_message_v4(row["text"], row["ste"], row["mode"], lex, proper_names(row["text"]))
            body = {"model": "judge", "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}],
                    "temperature": 0.0, "seed": 42, "max_tokens": args.max_tokens, "chat_template_kwargs": kwargs,
                    "id_slot": slot, "cache_prompt": True}
            reply = post(args.url, "/v1/chat/completions", body, 1800)["choices"][0]["message"].get("content") or ""
            try:
                differences = parse_judge(reply)
            except Exception:
                differences = None
            record = {"id": row["id"], "judge": args.model_id, "differences": differences, "raw_reply": reply}
            with lock:
                append(out / "second_judge.jsonl", record)
            return record
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(check, rows))
    flagged = sum(1 for r in results if r["differences"] is None or r["differences"])
    print(json.dumps({"run": args.run, "checked": len(results), "flagged": flagged}), flush=True)


if __name__ == "__main__":
    main()
