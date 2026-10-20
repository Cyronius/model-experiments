"""Teacher bake-off: run the same fixed prompts through candidate teacher models
and emit a side-by-side Markdown table for judging.

1. Select the fixed prompt set (once):
    python scripts/bakeoff.py select --input data/03_sampled.jsonl --output data/bakeoff_prompts.jsonl
2. Generate with each candidate (load the model in LM Studio first, repeat per model):
    python scripts/bakeoff.py run --model <lmstudio-model-id> [--no-think]
3. Build the comparison table:
    python scripts/bakeoff.py report --output data/bakeoff_report.md
"""

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from common.lmstudio_client import generate_title, warmup

PROMPTS_PATH = Path("data/bakeoff_prompts.jsonl")
RESULTS_DIR = Path("data/bakeoff_results")
PER_BUCKET = 10
BUCKETS = ["code_paste", "log_paste", "very_short", "vague", "long", "normal"]


def select(args) -> None:
    by_bucket = defaultdict(list)
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_bucket[rec["bucket"]].append(rec)
    rng = random.Random(7)
    picked = []
    for b in BUCKETS:
        pool = by_bucket.get(b, [])
        rng.shuffle(pool)
        picked.extend(pool[:PER_BUCKET])
        if len(pool) < PER_BUCKET:
            print(f"warning: bucket {b} has only {len(pool)} prompts")
    with Path(args.output).open("w", encoding="utf-8") as f:
        for rec in picked:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"selected {len(picked)} bake-off prompts -> {args.output}")


async def run(args) -> None:
    prompts = [json.loads(l) for l in PROMPTS_PATH.open(encoding="utf-8") if l.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_").replace(":", "_")
    out_path = RESULTS_DIR / f"{safe_name}.jsonl"
    sem = asyncio.Semaphore(4)

    async with httpx.AsyncClient() as client:
        await warmup(client, args.model)
        fout = out_path.open("w", encoding="utf-8")

        async def work(rec):
            async with sem:
                try:
                    title, attempts, reason = await generate_title(
                        client, rec["prompt"], args.model, no_think=args.no_think
                    )
                except httpx.HTTPError as e:
                    title, attempts, reason = None, 0, str(e)
                return {**{k: rec[k] for k in ("id", "bucket", "prompt")}, "title": title, "attempts": attempts, "fail_reason": reason}

        results = await asyncio.gather(*(work(r) for r in prompts))
        for row in results:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.close()
    ok = sum(1 for r in results if r["title"])
    print(f"{args.model}: {ok}/{len(results)} valid titles -> {out_path}")


JUDGE_SYSTEM = (
    "You rate how well a title describes the user message that starts a chat conversation. "
    "Score 1-5: 5 = accurate and specific, 3 = related but generic, 1 = wrong or misleading. "
    "Answer with ONLY the digit."
)


async def judge(args) -> None:
    import re

    from common.lmstudio_client import _post_with_retry, truncate_prompt

    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    sem = asyncio.Semaphore(2)  # big thinking judges are slow; more concurrency just times out
    async with httpx.AsyncClient() as client:
        await warmup(client, args.model)

        async def score(rec) -> int | None:
            if not rec["title"]:
                return 1  # failed generation scores worst
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"User message:\n<message>\n{truncate_prompt(rec['prompt'])}\n</message>\n\nTitle: {rec['title']}\n\nScore 1-5:",
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 4096,  # thinking models burn tokens before answering
            }
            async with sem:
                resp = await _post_with_retry(client, body, timeout=600)
            msg = resp.json()["choices"][0]["message"]
            text = msg.get("content") or ""
            text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
            m = re.search(r"[1-5]", text)
            if not m:  # answer never left the thinking block — take its last digit
                thinking = msg.get("reasoning_content") or ""
                digits = re.findall(r"[1-5]", thinking[-200:])
                return int(digits[-1]) if digits else None
            return int(m.group())

        print(f"judge: {args.model}")
        for path in files:
            rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            scores = [s for s in await asyncio.gather(*(score(r) for r in rows)) if s is not None]
            by_bucket = defaultdict(list)
            for r, s in zip(rows, scores):
                by_bucket[r["bucket"]].append(s)
            bucket_str = "  ".join(f"{b}={sum(v)/len(v):.2f}" for b, v in sorted(by_bucket.items()))
            print(f"{path.stem}: mean {sum(scores)/len(scores):.2f}  |  {bucket_str}")


def report(args) -> None:
    models = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        rows = {r["id"]: r for r in (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}
        models[path.stem] = rows
    if not models:
        print("no results in", RESULTS_DIR)
        return
    prompts = [json.loads(l) for l in PROMPTS_PATH.open(encoding="utf-8") if l.strip()]
    names = list(models)

    lines = ["# Teacher bake-off", "", "| bucket | prompt | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for rec in prompts:
        snippet = rec["prompt"][:80].replace("|", "\\|").replace("\n", " ")
        titles = [(models[n].get(rec["id"], {}).get("title") or "FAILED").replace("|", "\\|") for n in names]
        lines.append(f"| {rec['bucket']} | {snippet} | " + " | ".join(titles) + " |")

    lines += ["", "## Valid-title rate", ""]
    for n in names:
        ok = sum(1 for r in models[n].values() if r["title"])
        lines.append(f"- {n}: {ok}/{len(models[n])}")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sel = sub.add_parser("select")
    p_sel.add_argument("--input", required=True)
    p_sel.add_argument("--output", default=str(PROMPTS_PATH))
    p_run = sub.add_parser("run")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--no-think", action="store_true")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--output", default="data/bakeoff_report.md")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--model", required=True, help="large LM Studio model id used as judge")
    args = ap.parse_args()
    if args.cmd == "select":
        select(args)
    elif args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "judge":
        asyncio.run(judge(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
