"""Greedy vs beam decode comparison on the quantized ONNX artifact.

Decides the shipped decode settings for a model (see
.claude/plans/titler-v2-8x4-d320-gated.md): if greedy (beams=1) output is
byte-identical to beams=2 on nearly all inputs, ship greedy and pocket the
latency; otherwise ship beams=2.

Usage:
    python scripts/compare_decode.py --model model/titler-v2-8x4-web --test data/test.jsonl [--limit 500]

Prints disagreement rate + per-setting latency, and writes every differing
pair to data/decode_diff.md for eyeball review (greedy garbles are usually
obvious at a glance).
"""

import argparse
import json
import time
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

BASE_KWARGS = {"max_new_tokens": 32, "no_repeat_ngram_size": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="data/decode_diff.md")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForSeq2SeqLM.from_pretrained(args.model, subfolder="onnx", use_merged=True, use_cache=True)
    rows = [json.loads(l) for l in Path(args.test).open(encoding="utf-8") if l.strip()][: args.limit]

    results = {1: {"titles": [], "lat": []}, 2: {"titles": [], "lat": []}}
    for beams in (1, 2):
        for rec in rows:
            inputs = tokenizer(rec["prompt"], return_tensors="pt", truncation=True, max_length=512)
            start = time.perf_counter()
            out = model.generate(**inputs, num_beams=beams, **BASE_KWARGS)
            results[beams]["lat"].append((time.perf_counter() - start) * 1000)
            results[beams]["titles"].append(tokenizer.decode(out[0], skip_special_tokens=True).strip())

    n = len(rows)
    diffs = [
        (rec["prompt"], g, b)
        for rec, g, b in zip(rows, results[1]["titles"], results[2]["titles"])
        if g != b
    ]
    for beams in (1, 2):
        lat = sorted(results[beams]["lat"])
        print(f"beams={beams}: latency ms p50/p95 = {lat[n // 2]:.0f}/{lat[int(n * 0.95)]:.0f}")
    print(f"disagreements: {len(diffs)}/{n} ({len(diffs) / n:.1%})")

    lines = [
        "# Greedy (beams=1) vs beams=2 on the int8 model",
        "",
        f"{len(diffs)}/{n} differ. Review: is the greedy column garbled, or just different-but-fine?",
        "",
        "| prompt | greedy | beams=2 |",
        "|---|---|---|",
    ]
    for prompt, g, b in diffs:
        snippet = prompt[:70].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {snippet} | {g} | {b} |")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"diff report -> {args.out}")


if __name__ == "__main__":
    main()
