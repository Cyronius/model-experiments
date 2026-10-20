"""Stage 4: generate titles for sampled prompts via the LM Studio teacher.
Append-only and resumable: rerunning skips ids already present in the output.

Usage:
    python scripts/04_generate_titles.py --input data/03_sampled.jsonl --output data/04_titled.jsonl --model <lmstudio-model-id> [--no-think] [--concurrency 8] [--limit N] [--bucket code_paste]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from common.lmstudio_client import generate_title, warmup


async def run(args) -> None:
    done: set[str] = set()
    out_path = Path(args.output)
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"resuming: {len(done)} already titled")

    todo = []
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in done:
                continue
            if args.bucket and rec.get("bucket") != args.bucket:
                continue
            todo.append(rec)
            if args.limit and len(todo) >= args.limit:
                break
    print(f"to generate: {len(todo)}")
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "failed": 0, "start": time.time()}

    async with httpx.AsyncClient() as client:
        if not args.reasoning_effort:  # warmup is for LM Studio's model loading; OpenAI rejects its max_tokens
            await warmup(client, args.model)
        fout = out_path.open("a", encoding="utf-8")

        async def work(rec: dict) -> None:
            async with sem:
                try:
                    title, attempts, reason = await generate_title(
                        client, rec["prompt"], args.model, no_think=args.no_think, max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                    )
                except (httpx.HTTPError, KeyError) as e:
                    title, attempts, reason = None, 0, f"http error: {e}"
                row = {
                    "id": rec["id"],
                    "bucket": rec.get("bucket"),
                    "prompt": rec["prompt"],
                    "title": title,
                    "teacher": args.model,
                    "attempts": attempts,
                    "fail_reason": reason,
                }
                async with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["ok" if title else "failed"] += 1
                    n = stats["ok"] + stats["failed"]
                    if n % 100 == 0:
                        rate = n / (time.time() - stats["start"])
                        eta_h = (len(todo) - n) / rate / 3600 if rate else 0
                        print(f"{n}/{len(todo)}  {rate:.1f}/s  failed={stats['failed']}  eta={eta_h:.1f}h")

        await asyncio.gather(*(work(r) for r in todo))
        fout.close()

    print(f"done: {stats['ok']} ok, {stats['failed']} failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True, help="LM Studio model id (see /v1/models)")
    ap.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the default (1024 if --no-think else 24) -- use a small value "
                         "(e.g. 32-40) with --no-think on a backend confirmed to honor "
                         "chat_template_kwargs.enable_thinking (llama-server/vLLM, not LM Studio)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="OpenAI reasoning models: e.g. 'none' for gpt-6-luna (sends max_completion_tokens)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bucket", default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
