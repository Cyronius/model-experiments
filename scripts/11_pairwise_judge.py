"""Blind pairwise judge for the v2-vs-v1.5 eval (2026-10-17).

Each pair is judged in both orders; only verdicts that survive the swap count as a win/loss.
Usage: python scripts/11_pairwise_judge.py --a data/eval/titles_v2.json --b data/eval/titles_v15.json --judge <lmstudio-id>
       python scripts/11_pairwise_judge.py --calibrate --judge <lmstudio-id>
"""

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import httpx

from common.lmstudio_client import _post_with_retry, truncate_prompt, warmup

SYSTEM = (
    "You compare two candidate titles for a chat conversation, as shown in a sidebar. "
    "A good title names the topic of the user's message in 3-8 words, is fluent, and does not "
    "narrate ('User asks...') or copy the message. Answer with exactly one letter: "
    "A if title A is better, B if title B is better, or T if they are equally good. No other text."
)


async def ask(client, judge: str, prompt: str, a: str, b: str, extra: dict) -> str:
    extra = dict(extra)
    suffix = " /no_think" if extra.pop("no_think", False) else ""
    body = {
        "model": judge,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"User message:\n<message>\n{truncate_prompt(prompt)}\n</message>\n\n"
                                        f"Title A: {a}\nTitle B: {b}\n\nAnswer A, B, or T.{suffix}"},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        **extra,
    }
    if "reasoning_effort" in body:  # OpenAI reasoning models reject max_tokens
        body["max_completion_tokens"] = body.pop("max_tokens")
    resp = await _post_with_retry(client, body, timeout=600)
    text = re.sub(r"<think>.*?(</think>|$)", "", resp.json()["choices"][0]["message"]["content"] or "", flags=re.S)
    m = re.search(r"\b([ABT])\b", text.strip().upper())
    return m.group(1) if m else "T"


async def judge_pairs(judge: str, pairs: list[dict], extra: dict, concurrency: int = 4) -> list[str]:
    """Per pair: 'a', 'b' or 'tie' (tie also covers verdicts that flip with the order)."""
    sem = asyncio.Semaphore(concurrency)
    out = ["tie"] * len(pairs)
    async with httpx.AsyncClient() as client:
        if "reasoning_effort" not in extra:  # warmup is for LM Studio model loading
            await warmup(client, judge)

        async def one(i: int) -> None:
            p = pairs[i]
            async with sem:
                first, second = await asyncio.gather(
                    ask(client, judge, p["prompt"], p["a"], p["b"], extra),
                    ask(client, judge, p["prompt"], p["b"], p["a"], extra),
                )
            wins_a = (first == "A") + (second == "B")
            wins_b = (first == "B") + (second == "A")
            out[i] = "a" if wins_a > wins_b else "b" if wins_b > wins_a else "tie"

        await asyncio.gather(*(one(i) for i in range(len(pairs))))
    return out


def report(pairs: list[dict], verdicts: list[str]) -> None:
    c = Counter(verdicts)
    decided = c["a"] + c["b"]
    print(f"n={len(verdicts)}  A wins {c['a']}  B wins {c['b']}  tie/inconsistent {c['tie']}")
    if decided:
        print(f"A share of decided pairs: {c['a'] / decided:.1%}")
    for bucket in sorted({p["bucket"] for p in pairs}):
        bc = Counter(v for p, v in zip(pairs, verdicts) if p["bucket"] == bucket)
        print(f"  {bucket}: A {bc['a']}  B {bc['b']}  tie {bc['tie']}")


EVAL_SET = "data/eval/eval_set.jsonl"


def read_eval_set() -> list[dict]:
    with Path(EVAL_SET).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(a_path: str, b_path: str, limit: int | None, bucket: str | None = None) -> list[dict]:
    a, b = json.loads(Path(a_path).read_text(encoding="utf-8")), json.loads(Path(b_path).read_text(encoding="utf-8"))
    pairs = [{"id": r["id"], "bucket": r["bucket"], "prompt": r["prompt"], "a": a[r["id"]], "b": b[r["id"]]}
             for r in read_eval_set() if r["id"] in a and r["id"] in b and a[r["id"]] != b[r["id"]] and (not bucket or r["bucket"] == bucket)]
    return pairs[:limit] if limit else pairs


def calibration_pairs(n: int) -> list[dict]:
    """Teacher title (A) vs a title borrowed from an unrelated prompt (B): a sound judge picks A."""
    rows = [r for r in read_eval_set() if r["title"]][:n]
    rng = random.Random(3)
    return [{"id": r["id"], "bucket": "calib", "prompt": r["prompt"], "a": r["title"],
             "b": rng.choice([x for x in rows if x["id"] != r["id"]])["title"]} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bucket", help="only this bucket, e.g. hard")
    ap.add_argument("--eval-set", default=EVAL_SET, help="JSONL with id, bucket, prompt")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--extra", default="{}", help="JSON merged into the request, e.g. reasoning_effort")
    ap.add_argument("--dump", help="write per-pair verdicts here")
    args = ap.parse_args()
    globals()["EVAL_SET"] = args.eval_set
    pairs = calibration_pairs(args.limit or 100) if args.calibrate else load_pairs(args.a, args.b, args.limit, args.bucket)
    verdicts = asyncio.run(judge_pairs(args.judge, pairs, json.loads(args.extra)))
    report(pairs, verdicts)
    if args.dump:
        Path(args.dump).write_text(
            "".join(json.dumps({**p, "verdict": v}, ensure_ascii=False) + "\n" for p, v in zip(pairs, verdicts)),
            encoding="utf-8")


if __name__ == "__main__":
    main()
