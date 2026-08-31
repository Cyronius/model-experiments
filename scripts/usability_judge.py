"""Absolute-usability judge: rate each model's title alone (U/N) over the test set.

    python scripts/usability_judge.py --models model/titler-v1-web model/titler-33full-web model/titler-44full-web --judge granite-4.1-30b --limit 300
"""

import argparse
import asyncio
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import httpx

from common.lmstudio_client import _post_with_retry, truncate_prompt, warmup

warnings.filterwarnings("ignore")

SYSTEM = (
    "You rate whether a title is USABLE for a chat conversation that starts with the given user message. "
    "Usable = a reader scanning a sidebar would know roughly what the conversation is about; minor blandness is fine; "
    "garbled grammar, wrong topic, or nonsense words are not. Answer exactly U (usable) or N (not usable)."
)


def generate_all(model_dir: str, rows: list[dict]) -> list[str]:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, subfolder="onnx", use_merged=True, use_cache=True)
    out = []
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=512)
        gen = model.generate(**ids, max_new_tokens=32, num_beams=2, no_repeat_ngram_size=2)
        t = tok.decode(gen[0], skip_special_tokens=True).strip()
        t = re.sub(r"[\"'“”‘’`]", "", t)
        out.append(" ".join(t.split()[:8]))
    return out


async def judge_all(judge: str, rows: list[dict], titles: list[str]):
    sem = asyncio.Semaphore(4)
    verdicts = [0] * len(rows)
    async with httpx.AsyncClient() as client:
        await warmup(client, judge)

        async def one(i):
            body = {
                "model": judge, "temperature": 0.0, "max_tokens": 8,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"User message:\n<message>\n{truncate_prompt(rows[i]['prompt'])}\n</message>\n\nTitle: {titles[i]}\n\nU or N:"},
                ],
            }
            async with sem:
                resp = await _post_with_retry(client, body, timeout=600)
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip().upper()
            verdicts[i] = 1 if text.startswith("U") else 0

        await asyncio.gather(*(one(i) for i in range(len(rows))))
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path("data/test.jsonl").open(encoding="utf-8")][: args.limit]
    for model_dir in args.models:
        titles = generate_all(model_dir, rows)
        verdicts = asyncio.run(judge_all(args.judge, rows, titles))
        by = defaultdict(list)
        for r, v in zip(rows, verdicts):
            by[r["bucket"]].append(v)
        total = sum(verdicts)
        buckets = "  ".join(f"{b}={sum(v)}/{len(v)}" for b, v in sorted(by.items()))
        print(f"{model_dir}: {total}/{len(rows)} = {total/len(rows):.1%}  |  {buckets}", flush=True)


if __name__ == "__main__":
    main()
