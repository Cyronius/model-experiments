"""Score a meaning-check judge against assistant-labeled rewrites.

Reads data/out/doc-judge-eval/labeled.jsonl (source, rewrite, label kept/repair),
asks the judge served at --url for each pair with the v4 MEANING CHECK message and a
system prompt from ste_context_v5, and reports how many meaning changes it flags and
how many good rewrites it flags. A reply that cannot be parsed counts as a flag,
because the v4 pipeline rejects such a rewrite. The labels are the assistant's,
not owner-verified.
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
import time  # noqa: E402

sys.path[:0] = [str(Path(__file__).parent)]
from ste_context_v4 import judge_message_v4  # noqa: E402
from ste_context_v5 import judge_prompt  # noqa: E402
from ste_context_v3 import parse_judge  # noqa: E402
from synthesize_v2 import post  # noqa: E402
from synthesize_v5 import SLOTS, prepare_prefix  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/out/doc-judge-eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--kwargs", default='{"enable_thinking": false}', help="chat_template_kwargs as JSON")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--variant", default="core", help="judge prompt variant (ste_context_v5)")
    parser.add_argument("--model-id", required=True, help="the served model, for the cached prefix file")
    parser.add_argument("--min-n", type=int, default=0, help="only pairs from passage number n on (fresh pool: 1001)")
    args = parser.parse_args()
    lex = Lexicon()
    system = judge_prompt(lex, args.variant)
    pairs = [p for p in map(json.loads, (BASE / "labeled.jsonl").read_text().splitlines()) if p["n"] >= args.min_n]
    kwargs = json.loads(args.kwargs)
    SLOTS.mkdir(parents=True, exist_ok=True)
    prefix = prepare_prefix(args.url, system, {"kwargs": kwargs}, args.model_id, 7200)
    free, lock = list(range(args.parallel)), threading.Lock()

    def ask(pair):
        with lock:
            slot = free.pop()
        try:
            return _ask(pair, slot)
        finally:
            with lock:
                free.append(slot)

    def _ask(pair, slot):
        post(args.url, f"/slots/{slot}?action=restore", {"filename": prefix["file"]}, 600)
        message = judge_message_v4(pair["source"], pair["target"], pair["mode"], lex, proper_names(pair["source"]))
        body = {"model": "judge", "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}],
                "temperature": 0.0, "seed": 42, "max_tokens": args.max_tokens, "chat_template_kwargs": kwargs,
                "id_slot": slot, "cache_prompt": True}
        t0 = time.time()
        out = post(args.url, "/v1/chat/completions", body, 1800)
        reply = out["choices"][0]["message"]
        try:
            differences, error = parse_judge(reply.get("content") or ""), None
        except Exception as exc:
            differences, error = None, type(exc).__name__
        return {**pair, "differences": differences, "error": error, "raw_reply": reply.get("content"),
                "reasoning": reply.get("reasoning_content"), "seconds": round(time.time() - t0, 1),
                "usage": out.get("usage")}

    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(ask, pairs))
    wall = time.time() - t0
    flag = lambda r: r["differences"] is None or bool(r["differences"])
    changed = [r for r in rows if r["label"] == "repair"]
    kept = [r for r in rows if r["label"] == "kept"]
    out = BASE / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "judgements.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"name": args.name, "variant": args.variant, "kwargs": kwargs, "max_tokens": args.max_tokens,
               "pairs": len(rows), "wall_s": round(wall, 1),
               "meaning_changes_flagged": sum(map(flag, changed)), "of_meaning_changes": len(changed),
               "good_rewrites_flagged": sum(map(flag, kept)), "of_good_rewrites": len(kept),
               "unparsed": sum(r["differences"] is None for r in rows),
               "missed_n": sorted(r["n"] for r in changed if not flag(r)),
               "output_tokens": sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
