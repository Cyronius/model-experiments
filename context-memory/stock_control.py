#!/usr/bin/env python3
"""Original Qwen3-4B full-text and no-text controls, added after format preflight."""
import argparse
import datetime as dt
import json
import time

import torch
import transformers
from pathlib import Path

from memory import INSTRUCTION, answer, load_decoder, prefill, setup, sha256, text_embeddings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()
setup(args.threads)
args.output.mkdir(parents=True, exist_ok=True)
model, tokenizer = load_decoder(args.checkpoint)
questions = json.loads((args.data / "questions.json").read_text())
document = (args.data / "original.txt").read_text()
(args.output / "environment.json").write_text(json.dumps({
    "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "torch": torch.__version__, "transformers": transformers.__version__,
    "device": "cpu", "dtype": "bfloat16", "threads": args.threads,
    "greedy": True, "max_new_tokens": 40, "training_steps": 0,
    "source_sha256": sha256(document.encode()), "paid_compute_cost_usd": 0,
}, indent=2) + "\n")
prefixes = {}
with (args.output / "answers.jsonl").open("w") as stream:
    for condition in ("stock_no_memory", "stock_full_text"):
        content = document if condition == "stock_full_text" else "[No notes supplied.]"
        # Stock model has no learned memory-boundary tokens; use ordinary text.
        start = time.perf_counter()
        with torch.inference_mode():
            prefix = text_embeddings(model, tokenizer, "<|im_start|>user\n" + INSTRUCTION + content)
            cache = prefill(model, prefix)
        prefixes[condition] = {"prefix_positions": prefix.shape[1], "prefill_seconds": time.perf_counter() - start}
        print("PREFIX", condition, prefixes[condition], flush=True)
        for q in questions:
            result = answer(model, tokenizer, cache, q["question"], max_tokens=40)
            row = dict(q, condition=condition, **result)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(condition, q["id"], repr(row["answer"]), flush=True)
        del cache, prefix
(args.output / "prefixes.json").write_text(json.dumps(prefixes, indent=2) + "\n")
